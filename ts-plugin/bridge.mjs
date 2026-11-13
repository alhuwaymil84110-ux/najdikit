import { spawn } from "node:child_process";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const SRC_DIR = path.join(REPO_ROOT, "src");

/**
 * Persistent zero-dependency bridge to the Python `najdikit worker` process.
 * Keeps a warm worker alive so tool invocations complete in sub-millisecond to ~1ms latency.
 */
export class NajdiWorkerBridge {
  constructor(options = {}) {
    this.pythonBin = options.pythonBin || process.env.NAJDIKIT_PYTHON || "python3";
    this.defaultCustomLexicon = options.customLexiconPath || process.env.NAJDIKIT_CUSTOM_LEXICON || undefined;
    this.defaultPersonalAdapter = options.personalAdapterJson || process.env.NAJDIKIT_PERSONAL_ADAPTER_JSON || undefined;
    this.defaultLocalLlmUrl = options.localLlmUrl || process.env.NAJDIKIT_LOCAL_LLM_URL || undefined;
    this.proc = null;
    this.rl = null;
    this.seq = 0;
    this.pending = new Map();
  }

  _ensureWorker() {
    if (this.proc && !this.proc.killed && this.proc.exitCode === null) {
      return;
    }

    const existingPyPath = process.env.PYTHONPATH || "";
    const mergedPyPath = existingPyPath ? `${SRC_DIR}${path.delimiter}${existingPyPath}` : SRC_DIR;

    this.proc = spawn(this.pythonBin, ["-m", "najdikit", "worker"], {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        PYTHONPATH: mergedPyPath,
        PYTHONUTF8: "1",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    if (typeof this.proc.unref === "function") {
      this.proc.unref();
    }

    this.rl = readline.createInterface({
      input: this.proc.stdout,
      crlfDelay: Infinity,
    });

    this.rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let parsed;
      try {
        parsed = JSON.parse(trimmed);
      } catch {
        return;
      }
      const entry = this.pending.get(parsed.id);
      if (!entry) return;
      this.pending.delete(parsed.id);
      clearTimeout(entry.timer);
      if (parsed.ok) {
        entry.resolve(parsed.result);
      } else {
        entry.reject(new Error(parsed.error || "NajdiKit worker error"));
      }
    });

    const rejectAll = (err) => {
      for (const [, entry] of this.pending) {
        clearTimeout(entry.timer);
        entry.reject(err);
      }
      this.pending.clear();
      this.proc = null;
      this.rl = null;
    };

    this.proc.on("error", (err) => rejectAll(err));
    this.proc.on("exit", (code) => {
      if (this.pending.size > 0) {
        rejectAll(new Error(`NajdiKit worker exited unexpectedly (code=${code})`));
      } else {
        this.proc = null;
        this.rl = null;
      }
    });
  }

  call(action, params = {}, timeoutMs = 5000) {
    this._ensureWorker();
    const id = ++this.seq;
    const enrichedParams = {
      ...(this.defaultCustomLexicon && !params.custom_lexicon_path
        ? { custom_lexicon_path: this.defaultCustomLexicon }
        : {}),
      ...(this.defaultPersonalAdapter && !params.personal_adapter_json
        ? { personal_adapter_json: this.defaultPersonalAdapter }
        : {}),
      ...(this.defaultLocalLlmUrl && !params.local_llm_url
        ? { local_llm_url: this.defaultLocalLlmUrl }
        : {}),
      ...params,
    };

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`NajdiKit worker timed out after ${timeoutMs}ms on ${action}`));
      }, timeoutMs);
      if (typeof timer.unref === "function") {
        timer.unref();
      }
      this.pending.set(id, { resolve, reject, timer });
      const payload = JSON.stringify({ id, action, params: enrichedParams }) + "\n";
      this.proc.stdin.write(payload, "utf8");
    });
  }

  close() {
    if (this.proc && !this.proc.killed) {
      try {
        this.proc.stdin.end();
        this.proc.kill();
      } catch {
        // Ignore shutdown errors
      }
    }
    this.proc = null;
    this.rl = null;
  }
}

export const TOOL_DEFINITIONS = [
  {
    name: "najdi_tts_prepare",
    description:
      "Prepare Arabic or mixed Arabic/English text for natural Modern White Saudi & Urban Najdi TTS synthesis (applies Najdi imperfective kasrah, conversational pausal sukun, shaddah/madd accuracy, Saudi currency/number/time expansion, acronym pronunciation, and optional long-ID elision).",
    inputSchema: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Input text to convert into phonetically vocalized Saudi Arabic.",
        },
        skip_long_ids: {
          type: "boolean",
          description: "Omit unreadable UUIDs, URLs, long reference numbers, and phone numbers (default: true).",
        },
        expand_numbers: {
          type: "boolean",
          description: "Expand numbers, currencies (SAR/halalah/USD/%), dates, and times into spoken Saudi Arabic (default: true).",
        },
        expand_acronyms: {
          type: "boolean",
          description: "Expand common English tech acronyms like API, TTS, MCP into Arabic phonetics (default: true).",
        },
        use_mlx: {
          type: "boolean",
          description: "Use optional local MLX LLM server for contextual homograph disambiguation if available (default: false).",
        },
      },
      required: ["text"],
    },
  },
  {
    name: "najdi_stt_clean",
    description:
      "Post-process Arabic ASR/STT transcripts (strips Whisper/Scribe silence hallucinations, normalizes Modern White Saudi orthography consistently, optionally runs personal adapter hooks, and optionally returns carry prompts).",
    inputSchema: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Raw ASR/STT transcript text to clean and normalize.",
        },
        include_prompt: {
          type: "boolean",
          description: "Also return Whisper/Scribe `initial_prompt` and `carry_prompt` seeded from this transcript.",
        },
        domain: {
          type: "string",
          enum: ["general", "tech", "finance", "customer_care"],
          description: "Conversational domain when generating STT prompts.",
        },
        extra_terms: {
          type: "array",
          items: { type: "string" },
          description: "Optional domain vocabulary terms to boost in STT prompts.",
        },
      },
      required: ["text"],
    },
  },
  {
    name: "najdi_style_grounding",
    description:
      "Generate a positive, token-efficient System Prompt Grounding block so any LLM speaks authentic Modern White Saudi Arabic naturally without stiff MSA or regional dialect confusion.",
    inputSchema: {
      type: "object",
      properties: {
        profile: {
          type: "string",
          enum: ["modern_white", "executive", "voice_assistant", "technical"],
          description: "Target persona register (default: modern_white).",
        },
        compact: {
          type: "boolean",
          description: "Return compact low-token grounding block (default: false).",
        },
        domain_terms: {
          type: "array",
          items: { type: "string" },
          description: "Optional domain-specific vocabulary terms to include.",
        },
        harmonize_text: {
          type: "string",
          description: "Optional draft text to rewrite/harmonize if it contains non-Saudi or stiff-MSA drift.",
        },
      },
    },
  },
  {
    name: "najdi_lexicon_lookup",
    description:
      "Look up or search the 600+ entry Modern White Saudi & Urban Najdi lexicon, including vocalization, IPA hints, meanings, examples, and anti-confusion dialect checks.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Arabic word, phrase, English gloss, or non-Saudi term to look up.",
        },
        category: {
          type: "string",
          description: "Optional category filter (e.g. greetings_courtesy, verbs_desire_cognition, tech_workplace, finance_commerce).",
        },
        register: {
          type: "string",
          enum: ["universal", "executive", "casual", "technical"],
          description: "Optional register filter.",
        },
        limit: {
          type: "integer",
          description: "Maximum number of matching entries to return (default: 10).",
        },
      },
    },
  },
];

const ACTION_BY_TOOL = {
  najdi_tts_prepare: "tts_prepare",
  najdi_stt_clean: "stt_clean",
  najdi_style_grounding: "style_grounding",
  najdi_lexicon_lookup: "lexicon_lookup",
};

export async function dispatchToolCall(bridge, toolName, args = {}) {
  const action = ACTION_BY_TOOL[toolName];
  if (!action) {
    throw new Error(`Unknown NajdiKit tool: ${toolName}`);
  }
  return bridge.call(action, args || {});
}
