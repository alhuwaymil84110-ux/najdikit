#!/usr/bin/env node
/**
 * NajdiKit Standard Model Context Protocol (MCP) Server over stdio (JSON-RPC 2.0).
 * Zero external npm dependencies — compatible with Claude Desktop, Cursor, Antigravity, OpenClaw, and Codex.
 */

import { NajdiWorkerBridge, TOOL_DEFINITIONS, dispatchToolCall } from "./bridge.mjs";

const SERVER_INFO = {
  name: "najdikit-mcp",
  version: "2.0.0",
};

const DEFAULT_PROTOCOL_VERSION = "2024-11-05";
const bridge = new NajdiWorkerBridge();

let framingMode = "newline"; // Auto-switches to "content-length" if client sends LSP-style headers

function sendJsonRpc(message) {
  const serialized = JSON.stringify(message);
  if (framingMode === "content-length") {
    const byteLen = Buffer.byteLength(serialized, "utf8");
    process.stdout.write(`Content-Length: ${byteLen}\r\n\r\n${serialized}`);
  } else {
    process.stdout.write(`${serialized}\n`);
  }
}

function sendResult(id, result) {
  sendJsonRpc({ jsonrpc: "2.0", id, result });
}

function sendError(id, code, message, data = undefined) {
  sendJsonRpc({
    jsonrpc: "2.0",
    id,
    error: {
      code,
      message,
      ...(data !== undefined ? { data } : {}),
    },
  });
}

async function handleRpcMessage(msg) {
  if (!msg || typeof msg !== "object" || msg.jsonrpc !== "2.0") {
    sendError(msg?.id ?? null, -32600, "Invalid JSON-RPC 2.0 request");
    return;
  }

  const { id, method, params } = msg;
  const isNotification = id === undefined;

  try {
    if (method === "initialize") {
      const requestedVersion =
        typeof params?.protocolVersion === "string" ? params.protocolVersion : DEFAULT_PROTOCOL_VERSION;
      if (!isNotification) {
        sendResult(id, {
          protocolVersion: requestedVersion,
          capabilities: {
            tools: { listChanged: false },
          },
          serverInfo: SERVER_INFO,
        });
      }
      return;
    }

    if (method === "notifications/initialized" || method === "notifications/cancelled") {
      return;
    }

    if (method === "ping") {
      if (!isNotification) sendResult(id, {});
      return;
    }

    if (method === "tools/list") {
      if (!isNotification) {
        sendResult(id, {
          tools: TOOL_DEFINITIONS,
        });
      }
      return;
    }

    if (method === "tools/call") {
      const toolName = params?.name;
      const toolArgs = params?.arguments ?? {};
      if (!toolName || typeof toolName !== "string") {
        if (!isNotification) sendError(id, -32602, "Missing required tool `name` parameter");
        return;
      }

      try {
        const output = await dispatchToolCall(bridge, toolName, toolArgs);
        if (!isNotification) {
          sendResult(id, {
            content: [
              {
                type: "text",
                text: JSON.stringify(output, null, 2),
              },
            ],
            structuredContent: output,
            isError: false,
          });
        }
      } catch (toolErr) {
        if (!isNotification) {
          sendResult(id, {
            content: [
              {
                type: "text",
                text: String(toolErr?.message || toolErr),
              },
            ],
            isError: true,
          });
        }
      }
      return;
    }

    if (method === "resources/list") {
      if (!isNotification) sendResult(id, { resources: [] });
      return;
    }

    if (method === "prompts/list") {
      if (!isNotification) sendResult(id, { prompts: [] });
      return;
    }

    if (!isNotification) {
      sendError(id, -32601, `Method not found: ${method}`);
    }
  } catch (err) {
    if (!isNotification) {
      sendError(id, -32603, err?.message || "Internal MCP server error");
    }
  }
}

let rawBuffer = Buffer.alloc(0);
let rpcQueue = Promise.resolve();

function enqueueRpc(parsed) {
  rpcQueue = rpcQueue.then(() => handleRpcMessage(parsed)).catch(() => {});
}

function drainBuffer() {
  while (rawBuffer.length > 0) {
    const preview = rawBuffer.subarray(0, Math.min(rawBuffer.length, 32)).toString("utf8");

    if (preview.startsWith("Content-Length:")) {
      framingMode = "content-length";
      const headerEnd = rawBuffer.indexOf("\r\n\r\n");
      if (headerEnd === -1) return;
      const headerText = rawBuffer.subarray(0, headerEnd).toString("utf8");
      const match = /Content-Length:\s*(\d+)/i.exec(headerText);
      if (!match) {
        rawBuffer = rawBuffer.subarray(headerEnd + 4);
        continue;
      }
      const bodyLen = Number.parseInt(match[1], 10);
      const totalLen = headerEnd + 4 + bodyLen;
      if (rawBuffer.length < totalLen) return;
      const jsonText = rawBuffer.subarray(headerEnd + 4, totalLen).toString("utf8");
      rawBuffer = rawBuffer.subarray(totalLen);
      try {
        const parsed = JSON.parse(jsonText);
        enqueueRpc(parsed);
      } catch {
        sendError(null, -32700, "Parse error");
      }
      continue;
    }

    const newlineIdx = rawBuffer.indexOf(0x0a); // '\n'
    if (newlineIdx === -1) return;
    const lineText = rawBuffer.subarray(0, newlineIdx).toString("utf8").trim();
    rawBuffer = rawBuffer.subarray(newlineIdx + 1);
    if (!lineText) continue;

    framingMode = "newline";
    try {
      const parsed = JSON.parse(lineText);
      enqueueRpc(parsed);
    } catch {
      sendError(null, -32700, "Parse error");
    }
  }
}

process.stdin.on("data", (chunk) => {
  rawBuffer = Buffer.concat([rawBuffer, Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)]);
  drainBuffer();
});

process.stdin.on("end", () => {
  if (rawBuffer.length > 0) {
    const leftover = rawBuffer.toString("utf8").trim();
    rawBuffer = Buffer.alloc(0);
    if (leftover && !leftover.startsWith("Content-Length:")) {
      try {
        enqueueRpc(JSON.parse(leftover));
      } catch {
        sendError(null, -32700, "Parse error");
      }
    }
  }
  rpcQueue.finally(() => {
    bridge.close();
  });
});

process.on("SIGINT", () => {
  bridge.close();
  process.exit(0);
});

process.on("SIGTERM", () => {
  bridge.close();
  process.exit(0);
});
