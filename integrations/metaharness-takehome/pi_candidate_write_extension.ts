import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

type ExtensionAPI = { registerTool(tool: any): void; on?(event: string, handler: (event: any) => any): void };

// The only writable location: fixed names under the session workspace.
const OUT = resolve(process.cwd(), "out");
const MAX_CODE_CHARS = 60000;
// OpenEvolve's llm.max_tokens; without it DeepSeek truncates each turn at its 64K default.
const MAX_OUTPUT_TOKENS = 128000;

export default function (pi: ExtensionAPI) {
	pi.on?.("before_provider_request", (event) =>
		event?.payload && event.payload.max_tokens === undefined
			? { ...event.payload, max_tokens: MAX_OUTPUT_TOKENS }
			: undefined,
	);
	pi.registerTool({
		name: "write_candidate",
		label: "Write candidate program",
		description: "Submit the complete source of the one new candidate program, the id of the program it builds on, and a short change summary. Call exactly once.",
		parameters: {
			type: "object",
			required: ["code", "base_id", "changes"],
			properties: {
				code: { type: "string", minLength: 1, maxLength: MAX_CODE_CHARS },
				base_id: { type: "string", minLength: 1 },
				changes: { type: "string", minLength: 1 },
			},
			additionalProperties: false,
		},
		async execute(_toolCallId: string, params: any) {
			const code = params?.code;
			if (typeof code !== "string" || !code.trim() || code.length > MAX_CODE_CHARS) {
				throw new Error(`code must be a nonempty string of at most ${MAX_CODE_CHARS} characters`);
			}
			await mkdir(OUT, { recursive: true });
			await writeFile(resolve(OUT, "candidate.py"), code, "utf8");
			await writeFile(
				resolve(OUT, "pending.json"),
				JSON.stringify({ base_id: String(params.base_id), changes: String(params.changes) }) + "\n",
				"utf8",
			);
			return {
				content: [{ type: "text", text: "Candidate recorded. Stop now." }],
				details: { chars: code.length, base_id: String(params.base_id) },
			};
		},
	});
}
