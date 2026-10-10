/**
 * Resilient retry with staged backoff and ordered model fallback.
 *
 * Works automatically for whatever model is currently selected: when a
 * request fails with a transient error (rate limit, overload, 5xx,
 * timeout, connection drop), Pi settles the run and this extension takes
 * over at `agent_before_settle`:
 *
 *   retry 1: immediate, same model
 *   retry 2: sleep 5s, same model
 *   retry 3: sleep 10s, same model
 *   retry 4: sleep 10s, same model
 *   retry 5: sleep 20s, same model
 *   then: honor a server `retry-later` hint below 60s, and walk the
 *         fallback chain in order (one attempt per fallback):
 *
 *   a) opencode/step-5-preview-free              (high reasoning)
 *   b) openrouter/poolside/laguna-s-2.1:free     (highest reasoning)
 *   c) nvidia/nvidia/nemotron-3-ultra-550b-a55b
 *   d) nvidia/meta/muse-glimmer-30b
 *   e) nvidia/z-ai/glm-5.3-flash  (very high latency, often 1-2 min on finals)
 *
 * Non-transient errors (auth, quota/billing exhaustion, bad request,
 * context overflow) are never retried and fail fast, mirroring Pi's own
 * classifier. A successful response or a new user prompt resets the budget.
 * After a fallback switch, `/resilient-restore` returns to the original model.
 *
 * It also registers the virtual model `resilient/auto` (`/model resilient/auto`)
 * with the same schedule for anyone who prefers explicit opt-in. While a
 * virtual model is selected, the router owns retries and the automatic
 * handler stays out of the way.
 *
 * Recommended settings (merge into ~/.pi/agent/settings.json) so this
 * extension owns the exact schedule instead of Pi's exponential backoff:
 *
 *   {
 *     "retry": {
 *       "enabled": true,
 *       "maxRetries": 0,
 *       "baseDelayMs": 1000,
 *       "maxAgentDelayMs": 60000
 *     }
 *   }
 *
 * With the default `maxRetries: 3` everything still works: Pi's 3 backoff
 * retries run first, then this extension's 5 staged retries, then fallback.
 */

import type {
	ExtensionAPI,
	ExtensionContext,
	ModelRouteRequest,
} from "@earendil-works/pi-coding-agent";
import type { AssistantMessage, Model, ModelThinkingLevel } from "@earendil-works/pi-ai";

interface Fallback {
	provider: string;
	id: string;
	/** Preferred thinking level. `undefined` keeps the current level. */
	thinkingLevel?: ModelThinkingLevel;
	note: string;
}

const FALLBACKS: readonly Fallback[] = [
	{ provider: "opencode", id: "step-5-preview-free", thinkingLevel: "high", note: "high reasoning" },
	{ provider: "openrouter", id: "poolside/laguna-s-2.1:free", thinkingLevel: "xhigh", note: "highest reasoning" },
	{ provider: "nvidia", id: "nvidia/nemotron-3-ultra-550b-a55b", note: "fallback 3" },
	{ provider: "nvidia", id: "meta/muse-glimmer-30b", note: "fallback 4" },
	{ provider: "nvidia", id: "z-ai/glm-5.3-flash", note: "final fallback, very high latency (often 1-2 min on finals)" },
];

/** Extra sleep before same-model retries 1..5 (ms). Index 0 = first retry. */
const SAME_MODEL_DELAYS_MS = [0, 5_000, 10_000, 10_000, 20_000] as const;
const MAX_SAME_MODEL_RETRIES = SAME_MODEL_DELAYS_MS.length;
/** Server-requested delays at or above this are ignored (Pi default cap). */
const MAX_HONORED_RETRY_AFTER_MS = 60_000;
/** Pi API id of virtual-model catalog entries (see pi virtual-models.ts). */
const VIRTUAL_MODEL_API = "pi-virtual";

/*
 * Transient-error classifier. Mirrors Pi's own
 * `isRetryableAssistantError` (pi-ai utils/retry.ts), which extensions
 * cannot import, so quota/billing/auth errors fail fast here exactly like
 * they do in Pi's built-in retry.
 */
const NON_RETRYABLE_LIMIT_PATTERN = new RegExp(
	[
		"GoUsageLimitError",
		"FreeUsageLimitError",
		"Monthly usage limit reached",
		"available balance",
		"insufficient_quota",
		"out of budget",
		"quota exceeded",
		"billing",
		"subscription_sharing_usage_limit_exceeded",
	].join("|"),
	"i",
);
const RETRYABLE_ERROR_PATTERN = new RegExp(
	[
		"overloaded",
		"server_busy",
		"servers are currently busy",
		"currently experiencing high demand",
		"model is at capacity",
		"rate.?limit",
		"too many requests",
		"429",
		"500",
		"502",
		"503",
		"504",
		"520",
		"524",
		"service.?unavailable",
		"server.?error",
		"internal.?error",
		"provider.?returned.?error",
		"exceeded request buffer limit while retrying upstream",
		"network.?error",
		"connection.?error",
		"connection.?refused",
		"connection.?lost",
		"other side closed",
		"fetch failed",
		"getaddrinfo",
		"ENOTFOUND",
		"EAI_AGAIN",
		"upstream.?connect",
		"reset before headers",
		"socket hang up",
		"socket connection was closed",
		"timed? out",
		"timeout",
		"terminated",
		"websocket.?closed",
		"websocket.?error",
		"ended without",
		"stream ended before message_stop",
		"stream ended before a terminal response event",
		"http2 request did not get a response",
		"pending stream has been canceled",
		"retry delay",
		"you can retry your request",
		"try your request again",
		"please retry your request",
		"ResourceExhausted",
		"subscription_sharing_usage_unavailable",
		"subscription_sharing_user_unavailable",
	].join("|"),
	"i",
);
/** Context overflow is handled by compaction, never by retry. */
const CONTEXT_OVERFLOW_PATTERN =
	/context.?overflow|context_length_exceeded|too many tokens|token.?limit|prompt too (long|large)|maximum context|context window/i;

function isRetryableErrorMessage(errorMessage: string): boolean {
	if (!errorMessage) return false;
	if (NON_RETRYABLE_LIMIT_PATTERN.test(errorMessage)) return false;
	if (CONTEXT_OVERFLOW_PATTERN.test(errorMessage)) return false;
	return RETRYABLE_ERROR_PATTERN.test(errorMessage);
}

/**
 * Parse a server-requested retry delay from a provider error message.
 * Returns milliseconds, or 0 when no usable hint is present.
 */
export function parseRetryAfterMs(errorMessage: string): number {
	if (!errorMessage) return 0;
	const text = errorMessage.toLowerCase();

	// Explicit seconds: "retry after 45s", "retry-after: 45", "retry in 45 seconds".
	const secPatterns = [
		/retry[-_ ]after\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(m?s(?:ec(?:ond)?s?)?)?/,
		/retry\s+in\s+(\d+(?:\.\d+)?)\s*(m?s(?:ec(?:ond)?s?)?|min(?:ute)?s?)?/,
		/try\s+again\s+in\s+(\d+(?:\.\d+)?)\s*(m?s(?:ec(?:ond)?s?)?|min(?:ute)?s?)?/,
		/wait\s+(\d+(?:\.\d+)?)\s*seconds?\s+(?:before\s+)?retry/,
	];
	for (const pattern of secPatterns) {
		const match = pattern.exec(text);
		if (!match) continue;
		const value = Number(match[1]);
		if (!Number.isFinite(value) || value <= 0) continue;
		const unit = (match[2] ?? "s").trim();
		if (unit.startsWith("min")) return value * 60_000;
		if (unit.startsWith("ms")) return value;
		return value * 1_000;
	}

	// Millisecond hints: "retry in 30000ms".
	const msMatch = /retry\s+in\s+(\d+)\s*ms/.exec(text);
	if (msMatch) {
		const value = Number(msMatch[1]);
		if (Number.isFinite(value) && value > 0) return value;
	}
	return 0;
}

function abortableSleep(ms: number, signal?: AbortSignal): Promise<void> {
	if (ms <= 0) return Promise.resolve();
	return new Promise((resolve, reject) => {
		if (signal?.aborted) {
			reject(new Error("aborted"));
			return;
		}
		const timer = setTimeout(() => {
			signal?.removeEventListener("abort", onAbort);
			resolve();
		}, ms);
		const onAbort = () => {
			clearTimeout(timer);
			reject(new Error("aborted"));
		};
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

function modelKey(provider: string, id: string): string {
	return `${provider}/${id}`;
}

function sameModel(a: { provider: string; id: string }, b: { provider: string; id: string }): boolean {
	return a.provider === b.provider && a.id === b.id;
}

function lookupFallback(ctx: ExtensionContext, fallback: Fallback): Model | undefined {
	try {
		return ctx.modelRegistry.find(fallback.provider, fallback.id) ?? undefined;
	} catch {
		return undefined;
	}
}

function notify(ctx: ExtensionContext, message: string): void {
	try {
		ctx.ui.notify(message, "info");
	} catch {
		// UI is unavailable for background/direct requests; routing continues.
	}
}

/** Consecutive-failure episode for the automatic (non-virtual) path. */
interface AutoErrorInfo {
	entryId: string;
	provider: string;
	model: string;
	errorMessage: string;
}

const auto: {
	/** provider/id of the model currently being nursed through failures. */
	currentKey?: string;
	/** Same-model retries served in the primary phase. */
	primaryRetries: number;
	/** Index into FALLBACKS of the active fallback, -1 while on primary. */
	fallbackIdx: number;
	/** Latest turn_end failure, consumed by agent_before_settle. */
	stash?: AutoErrorInfo;
	/** Model the user had before the first automatic fallback switch. */
	switchedFrom?: { provider: string; id: string };
} = { primaryRetries: 0, fallbackIdx: -1 };

function resetEpisode(): void {
	auto.currentKey = undefined;
	auto.primaryRetries = 0;
	auto.fallbackIdx = -1;
	auto.stash = undefined;
}

export default function (pi: ExtensionAPI) {
	// ------------------------------------------------------------------
	// Automatic path: track failures, take over at agent_before_settle.
	// ------------------------------------------------------------------

	pi.on("turn_end", async (event) => {
		if (event.outcome === "error" && event.message.role === "assistant") {
			const message = event.message as AssistantMessage;
			auto.stash = {
				entryId: event.messageEntryId,
				provider: message.provider,
				model: message.model,
				errorMessage: message.errorMessage ?? "",
			};
		} else if (event.outcome === "completed") {
			// A good response ends any failure episode.
			resetEpisode();
		} else {
			auto.stash = undefined;
		}
		return undefined;
	});

	pi.on("before_agent_start", async () => {
		// A fresh user prompt gets a fresh retry budget.
		resetEpisode();
		return undefined;
	});

	pi.on("agent_before_settle", async (event, ctx) => {
		try {
			return await handleAutoFallback(event, ctx);
		} catch (error) {
			notify(ctx, `[resilient] auto-fallback skipped: ${error instanceof Error ? error.message : String(error)}`);
			resetEpisode();
			return undefined;
		}
	});

	async function handleAutoFallback(
		event: { outcome: string; context: { canContinue: boolean; contextEntries: { sourceEntry: { id: string } }[] } },
		ctx: ExtensionContext,
	) {
		if (event.outcome !== "error") {
			resetEpisode();
			return undefined;
		}
		// Virtual selections (incl. resilient/auto) are owned by their router.
		if (ctx.model && (ctx.model as { api?: string }).api === VIRTUAL_MODEL_API) return undefined;

		const failure = auto.stash;
		auto.stash = undefined;
		if (!failure) return undefined;
		if (!isRetryableErrorMessage(failure.errorMessage)) {
			resetEpisode();
			return undefined;
		}

		const key = modelKey(failure.provider, failure.model);
		if (key !== auto.currentKey) {
			// New failure episode (or the model changed under us).
			auto.currentKey = key;
			auto.primaryRetries = 0;
			auto.fallbackIdx = -1;
		}

		// Draft omitting the failed attempt so the continuation has runnable
		// context (mirrors Pi's own _omitRecoveryAttempt). Empty when Pi's
		// built-in retry already omitted it.
		const targetPresent = event.context.contextEntries.some(
			(entry) => entry.sourceEntry.id === failure.entryId,
		);
		const omitEntries = targetPresent
			? [{ type: "context_edit" as const, targetId: failure.entryId, replacement: null }]
			: [];
		if (!targetPresent && !event.context.canContinue) return undefined;

		if (auto.fallbackIdx === -1 && auto.primaryRetries < MAX_SAME_MODEL_RETRIES) {
			auto.primaryRetries += 1;
			const delayMs = SAME_MODEL_DELAYS_MS[auto.primaryRetries - 1];
			if (delayMs > 0) {
				notify(
					ctx,
					`[resilient] retry ${auto.primaryRetries}/${MAX_SAME_MODEL_RETRIES} on ${key} after ${delayMs / 1000}s`,
				);
				await abortableSleep(delayMs, ctx.signal ?? undefined);
			}
			return { entries: omitEntries, continue: true };
		}

		// Same-model budget exhausted, or our fallback just failed:
		// honor a server retry-later hint below 60s, then advance the chain.
		const hintedMs = parseRetryAfterMs(failure.errorMessage);
		if (hintedMs > 0 && hintedMs < MAX_HONORED_RETRY_AFTER_MS) {
			notify(ctx, `[resilient] server asked to retry later; waiting ${Math.round(hintedMs / 1000)}s`);
			await abortableSleep(hintedMs, ctx.signal ?? undefined);
		}

		const exclude = { provider: failure.provider, id: failure.model };
		for (let idx = auto.fallbackIdx + 1; idx < FALLBACKS.length; idx += 1) {
			const candidate = FALLBACKS[idx];
			if (sameModel(candidate, exclude)) continue;
			const model = lookupFallback(ctx, candidate);
			if (!model) continue; // Not in catalog or no credentials: try next.
			if (!(await pi.setModel(model))) continue;
			if (candidate.thinkingLevel) pi.setThinkingLevel(candidate.thinkingLevel as never);
			if (auto.fallbackIdx === -1) auto.switchedFrom = { ...exclude };
			auto.fallbackIdx = idx;
			auto.currentKey = modelKey(candidate.provider, candidate.id);
			notify(ctx, `[resilient] switching to fallback ${auto.currentKey} (${candidate.note})`);
			return { entries: omitEntries, continue: true };
		}

		notify(ctx, "[resilient] fallback chain exhausted; surfacing the error");
		resetEpisode();
		return undefined;
	}

	// ------------------------------------------------------------------
	// Explicit path: virtual model resilient/auto (unchanged behavior).
	// ------------------------------------------------------------------

	pi.registerVirtualModel<{ sameModelRetries: number; fallbackIdx: number }>({
		provider: "resilient",
		id: "auto",
		name: "Auto (resilient retry + fallback)",
		thinkingLevels: ["low", "medium", "high", "xhigh"],
		contextWindow: 200_000,
		maxTokens: 32_000,
		async route(request, ctx) {
			const state = request.state ?? { sameModelRetries: 0, fallbackIdx: -1 };

			// Non-retry requests: stay sticky on the last good model.
			if (request.reason !== "retry") {
				const sticky = request.previous;
				if (sticky) {
					return {
						model: sticky.model,
						thinkingLevel: sticky.thinkingLevel ?? request.thinkingLevel,
						state,
					};
				}
				const first = FALLBACKS[0];
				const model = lookupFallback(ctx, first);
				if (!model) throw new Error(`[resilient] primary fallback ${first.provider}/${first.id} is not in the catalog`);
				return { model, thinkingLevel: first.thinkingLevel ?? request.thinkingLevel, state };
			}

			// Retry without failure details (router itself failed earlier):
			// keep the last good model when one exists.
			if (!request.failed) {
				if (request.previous) {
					return {
						model: request.previous.model,
						thinkingLevel: request.previous.thinkingLevel ?? request.thinkingLevel,
						state,
					};
				}
				const first = FALLBACKS[0];
				const model = lookupFallback(ctx, first);
				if (!model) throw new Error("[resilient] no previous model and no fallback available");
				return { model, thinkingLevel: first.thinkingLevel ?? request.thinkingLevel, state };
			}

			const failed = request.failed;
			const errorMessage = failed.message.errorMessage ?? "";

			// Phase 1: staged same-model retries (1 immediate, then 5s/10s/10s/20s).
			if (state.sameModelRetries < MAX_SAME_MODEL_RETRIES) {
				state.sameModelRetries += 1;
				const delayMs = SAME_MODEL_DELAYS_MS[state.sameModelRetries - 1];
				if (delayMs > 0) {
					notify(
						ctx,
						`[resilient] retry ${state.sameModelRetries}/${MAX_SAME_MODEL_RETRIES} on ${failed.model.provider}/${failed.model.id} after ${delayMs / 1000}s (${errorMessage.slice(0, 120)})`,
					);
					await abortableSleep(delayMs, request.signal ?? undefined).catch(() => undefined);
				}
				return {
					model: failed.model,
					thinkingLevel: failed.thinkingLevel ?? request.thinkingLevel,
					state,
				};
			}

			// Phase 2: honor a server retry-later hint below 60s, then fall back.
			const hintedMs = parseRetryAfterMs(errorMessage);
			if (hintedMs > 0 && hintedMs < MAX_HONORED_RETRY_AFTER_MS) {
				notify(ctx, `[resilient] server asked to retry later; waiting ${Math.round(hintedMs / 1000)}s`);
				await abortableSleep(hintedMs, request.signal ?? undefined).catch(() => undefined);
			}

			for (let idx = state.fallbackIdx + 1; idx < FALLBACKS.length; idx += 1) {
				const candidate = FALLBACKS[idx];
				if (sameModel(candidate, { provider: failed.model.provider, id: failed.model.id })) {
					state.fallbackIdx = idx;
					continue;
				}
				const model = lookupFallback(ctx, candidate);
				if (!model) continue; // No credentials or not in catalog: try next.
				state.fallbackIdx = idx;
				state.sameModelRetries = 0;
				notify(
					ctx,
					`[resilient] switching to fallback ${candidate.provider}/${candidate.id} (${candidate.note})`,
				);
				return { model, thinkingLevel: candidate.thinkingLevel ?? request.thinkingLevel, state };
			}

			// Chain exhausted: stay on the failed model so Pi surfaces its error.
			notify(ctx, "[resilient] fallback chain exhausted; keeping failed model for final error");
			return {
				model: failed.model,
				thinkingLevel: failed.thinkingLevel ?? request.thinkingLevel,
				state,
			};
		},
	});

	pi.registerCommand("resilient-info", {
		description: "Show resilient retry schedule, fallback chain, and availability",
		handler: async (_args, ctx) => {
			const lines = [
				"Automatic for the active model: immediate, 5s, 10s, 10s, 20s; then server retry-later <60s; then fallback chain.",
				`Episode: ${auto.currentKey ?? "none"} (same-model retries: ${auto.primaryRetries}, fallback: ${auto.fallbackIdx >= 0 ? FALLBACKS[auto.fallbackIdx].id : "none"})`,
				`Switched from: ${auto.switchedFrom ? modelKey(auto.switchedFrom.provider, auto.switchedFrom.id) : "none"} (restore with /resilient-restore)`,
				"Fallback chain:",
			];
			for (const fallback of FALLBACKS) {
				const found = lookupFallback(ctx, fallback);
				lines.push(`  ${found ? "[ok]" : "[missing]"} ${fallback.provider}/${fallback.id} (${fallback.note})`);
			}
			lines.push("Recommended: retry.maxRetries = 0 so this extension owns the exact schedule.");
			ctx.ui.notify(lines.join("\n"), "info");
		},
	});

	pi.registerCommand("resilient-restore", {
		description: "Return to the model active before the automatic fallback switch",
		handler: async (_args, ctx) => {
			if (!auto.switchedFrom) {
				ctx.ui.notify("[resilient] no automatic fallback switch to restore", "info");
				return;
			}
			const target = ctx.modelRegistry.find(auto.switchedFrom.provider, auto.switchedFrom.id) ?? undefined;
			if (!target || !(await pi.setModel(target))) {
				ctx.ui.notify(
					`[resilient] cannot restore ${modelKey(auto.switchedFrom.provider, auto.switchedFrom.id)} (not in catalog or no credentials)`,
					"info",
				);
				return;
			}
			resetEpisode();
			ctx.ui.notify(`[resilient] restored ${modelKey(auto.switchedFrom.provider, auto.switchedFrom.id)}`, "info");
		},
	});
}
