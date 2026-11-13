import { NajdiWorkerBridge, TOOL_DEFINITIONS, dispatchToolCall } from "./bridge.mjs";

function formatOpenClawToolResult(payload) {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(payload, null, 2),
      },
    ],
    details: payload,
  };
}

/**
 * OpenClaw Plugin Entrypoint for NajdiKit.
 * Registers `najdi_tts_prepare`, `najdi_stt_clean`, `najdi_style_grounding`, and `najdi_lexicon_lookup`.
 */
const najdikitPlugin = {
  id: "najdikit",
  name: "NajdiKit",
  description:
    "Modern White Saudi & Urban Najdi Arabic phonetics (TTS), STT post-processing, lexicon lookup, and positive LLM grounding.",

  register(api) {
    const pluginConfig = (api && (api.pluginConfig || api.config?.plugins?.entries?.najdikit?.config)) || {};
    const bridge = new NajdiWorkerBridge({
      pythonBin: pluginConfig.pythonBin,
      customLexiconPath: pluginConfig.customLexiconPath,
      personalAdapterJson: pluginConfig.personalAdapterJson,
      localLlmUrl: pluginConfig.localLlmUrl,
    });

    for (const def of TOOL_DEFINITIONS) {
      api.registerTool({
        name: def.name,
        description: def.description,
        parameters: def.inputSchema,
        async execute(_id, params = {}) {
          const mergedParams =
            def.name === "najdi_style_grounding" && !params.profile && pluginConfig.defaultProfile
              ? { ...params, profile: pluginConfig.defaultProfile }
              : params;
          const result = await dispatchToolCall(bridge, def.name, mergedParams);
          return formatOpenClawToolResult(result);
        },
      });
    }

    if (typeof api.onShutdown === "function") {
      api.onShutdown(() => bridge.close());
    }
  },
};

export { NajdiWorkerBridge, TOOL_DEFINITIONS, dispatchToolCall };
export default najdikitPlugin;
