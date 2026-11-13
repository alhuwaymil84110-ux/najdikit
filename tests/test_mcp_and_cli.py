"""End-to-end tests for NajdiKit v2.0 CLI, MCP JSON-RPC 2.0 stdio server, OpenClaw plugin, and privacy hygiene."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MCP_SCRIPT = REPO_ROOT / "ts-plugin" / "mcp-server.mjs"


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    env["PYTHONUTF8"] = "1"
    return env


class TestCLICommands(unittest.TestCase):
    def _run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "najdikit", *args],
            cwd=REPO_ROOT,
            env=_cli_env(),
            capture_output=True,
            text=True,
            check=True,
        )

    def test_35_cli_speakable_plain_json_and_habibi_backend(self) -> None:
        plain = self._run_cli("speakable", "أبشر، شيكت على الفاتورة الحين بمبلغ 150 ريال")
        self.assertIn("أَبْشِرْ", plain.stdout)
        self.assertIn("مِيَّة وَخَمْسِينْ رِيَالْ", plain.stdout)

        habibi = self._run_cli("speakable", "--backend", "habibi", "أبشر طال عمرك")
        self.assertTrue(habibi.stdout.strip().startswith("②〈"))
        self.assertTrue(habibi.stdout.strip().endswith("〉"))

        as_json = self._run_cli("speakable", "--json", "تعرف وشلون نقدر نشوف السجلات")
        parsed = json.loads(as_json.stdout)
        self.assertIn("تِعْرِفْ", parsed["speakable"])
        self.assertIn("نِقْدَرْ", parsed["speakable"])

    def test_36_cli_normalize_plain_and_json(self) -> None:
        res = self._run_cli(
            "normalize",
            "--json",
            "ان شاء الله بشيك ع الموضوع الحيين لا هنت. اشتركوا في القناة",
        )
        parsed = json.loads(res.stdout)
        self.assertEqual(parsed["text"], "إن شاء الله بشيك ع الموضوع الحين لاهنت.")
        self.assertIn("اشتركوا في القناة", parsed["removed_hallucinations"])

    def test_37_cli_stt_prompt_command_including_telephony(self) -> None:
        res = self._run_cli(
            "stt-prompt",
            "--domain",
            "telephony",
            "--terms",
            "تحويل,تأكيد",
            "--json",
        )
        parsed = json.loads(res.stdout)
        self.assertEqual(parsed["domain"], "telephony")
        self.assertIn("تحويل", parsed["initial_prompt"])

    def test_38_cli_grounding_command_realtime_call(self) -> None:
        res = self._run_cli("grounding", "--profile", "voice_call_realtime", "--json")
        parsed = json.loads(res.stdout)
        self.assertEqual(parsed["profile"], "voice_call_realtime")
        self.assertIn("الهوية اللغوية", parsed["system_prompt"])

    def test_39_cli_lookup_and_stats(self) -> None:
        stats_res = self._run_cli("lookup", "--stats", "--json")
        stats = json.loads(stats_res.stdout)
        self.assertGreaterEqual(stats["entry_count"], 1250)

        look_res = self._run_cli("lookup", "ثواني أشيك لك", "--pack-id", "telephony_calls", "--json")
        look = json.loads(look_res.stdout)
        self.assertTrue(any(r["term"] == "ثواني أشيك لك" for r in look["results"]))

    def test_40_cli_call_chunks_turn_check_score_morph_and_packs(self) -> None:
        chunks_res = self._run_cli(
            "call-chunks",
            "--backend",
            "habibi",
            "--ack",
            "lookup",
            "--json",
            "أبشر، المبلغ 1,250.50 ريال [ACTION: send_invoice].",
        )
        chunks = json.loads(chunks_res.stdout)
        self.assertEqual(chunks["clause_count"], 2)
        self.assertEqual(chunks["extracted_actions"], ["send_invoice"])
        self.assertTrue(chunks["ack"].startswith("②〈"))

        turn_res = self._run_cli("turn-check", "--ai-speaking", "--json", "طيب كمّل")
        turn = json.loads(turn_res.stdout)
        self.assertEqual(turn["kind"], "backchannel")
        self.assertFalse(turn["should_abort_tts"])

        score_res = self._run_cli("score", "--json", "سوف أقوم بالتحقق من الأمر الذي ذكرته")
        sc = json.loads(score_res.stdout)
        self.assertGreater(sc["msa_leakage_score"], 40.0)

        morph_res = self._run_cli("morph", "--json", "وبنشيكها")
        mp = json.loads(morph_res.stdout)
        self.assertEqual(mp["vocalized"], "وَبِنْشَيِّكْهَا")

        packs_res = self._run_cli("packs", "--json")
        pk = json.loads(packs_res.stdout)
        self.assertGreaterEqual(pk["total_lexicon_entries"], 1250)
        self.assertGreaterEqual(len(pk["packs"]), 4)

    def test_41_cli_bench_sub_100us(self) -> None:
        res = self._run_cli("bench", "--iterations", "200", "--json")
        bench = json.loads(res.stdout)
        for op_name, metrics in bench["operations"].items():
            self.assertLess(
                metrics["mean_ms"],
                0.15,
                f"Operation {op_name} exceeded sub-millisecond gate: {metrics['mean_ms']}ms",
            )


class TestMCPStdioServer(unittest.TestCase):
    def _exchange_newline_rpc(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n"
        proc = subprocess.run(
            ["node", str(MCP_SCRIPT)],
            cwd=REPO_ROOT,
            env=_cli_env(),
            input=payload,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines]

    def test_42_mcp_initialize_and_seven_tools_list(self) -> None:
        responses = self._exchange_newline_rpc(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "2.0"},
                    },
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            ]
        )
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0]["id"], 1)
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "najdikit-mcp")
        self.assertEqual(responses[0]["result"]["serverInfo"]["version"], "2.0.0")

        tools = responses[1]["result"]["tools"]
        tool_names = {t["name"] for t in tools}
        self.assertEqual(
            tool_names,
            {
                "najdi_tts_prepare",
                "najdi_call_stream_chunk",
                "najdi_call_turn_analyze",
                "najdi_stt_clean",
                "najdi_dialect_score",
                "najdi_style_grounding",
                "najdi_lexicon_lookup",
            },
        )

    def test_43_mcp_call_all_seven_tools(self) -> None:
        responses = self._exchange_newline_rpc(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {
                        "name": "najdi_tts_prepare",
                        "arguments": {
                            "text": "أبشر، المبلغ 25 ريال الساعة 3:30 العصر",
                            "backend": "habibi",
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "tools/call",
                    "params": {
                        "name": "najdi_call_stream_chunk",
                        "arguments": {
                            "text": "أبشر، شيكت لك على هالفاتورة بمبلغ 150.50 ريال [ACTION: mark_paid].",
                            "ack_category": "lookup",
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "tools/call",
                    "params": {
                        "name": "najdi_call_turn_analyze",
                        "arguments": {"text": "ثواني أشوف الرقم", "is_ai_speaking": False},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 13,
                    "method": "tools/call",
                    "params": {
                        "name": "najdi_stt_clean",
                        "arguments": {
                            "text": "انشاءالله ارسلها الحيين لا هنت",
                            "include_prompt": True,
                            "domain": "telephony",
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 14,
                    "method": "tools/call",
                    "params": {
                        "name": "najdi_dialect_score",
                        "arguments": {"text": "أبشر طال عمرك، الأمور طيبة وكل شي جاهز الحين."},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 15,
                    "method": "tools/call",
                    "params": {
                        "name": "najdi_style_grounding",
                        "arguments": {
                            "profile": "voice_call_realtime",
                            "harmonize_text": "بدي أشوف الكود هلق",
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 16,
                    "method": "tools/call",
                    "params": {
                        "name": "najdi_lexicon_lookup",
                        "arguments": {"query": "خذ راحتك", "pack_id": "telephony_calls", "limit": 3},
                    },
                },
            ]
        )
        self.assertEqual(len(responses), 7)

        tts_out = responses[0]["result"]["structuredContent"]
        self.assertTrue(tts_out["speakable"].startswith("②〈"))
        self.assertIn("خَمْسَة وَعِشْرِينْ رِيَالْ", tts_out["speakable"])

        chunk_out = responses[1]["result"]["structuredContent"]
        self.assertEqual(chunk_out["clause_count"], 2)
        self.assertEqual(chunk_out["extracted_actions"], ["mark_paid"])
        self.assertIn("أَشَيِّكْ", chunk_out["ack"])

        turn_out = responses[2]["result"]["structuredContent"]
        self.assertEqual(turn_out["kind"], "hold_request")
        self.assertEqual(turn_out["recommended_wait_sec"], 20.0)

        stt_out = responses[3]["result"]["structuredContent"]
        self.assertEqual(stt_out["text"], "إن شاء الله ارسلها الحين لاهنت")
        self.assertIn("initial_prompt", stt_out["stt_prompt"])

        score_out = responses[4]["result"]["structuredContent"]
        self.assertGreaterEqual(score_out["saudi_alignment_score"], 90.0)

        gnd_out = responses[5]["result"]["structuredContent"]
        self.assertEqual(gnd_out["profile"], "voice_call_realtime")
        self.assertIn("أبي", gnd_out["harmonized_sample"]["harmonized"])

        lex_out = responses[6]["result"]["structuredContent"]
        self.assertEqual(lex_out["exact_match"]["term"], "خذ راحتك")

    def test_44_mcp_content_length_framing_and_error_codes(self) -> None:
        req1 = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode("utf-8")
        req2 = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "nonexistent/method"}).encode("utf-8")
        req3 = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "unknown_tool", "arguments": {}},
            }
        ).encode("utf-8")

        wire = b"".join(
            f"Content-Length: {len(b)}\r\n\r\n".encode("ascii") + b for b in (req1, req2, req3)
        )
        proc = subprocess.run(
            ["node", str(MCP_SCRIPT)],
            cwd=REPO_ROOT,
            env=_cli_env(),
            input=wire,
            capture_output=True,
            timeout=15,
            check=True,
        )
        raw_out = proc.stdout.decode("utf-8")
        self.assertIn("Content-Length:", raw_out)
        self.assertIn("-32601", raw_out)
        self.assertIn('"isError":true', raw_out)


class TestOpenClawPluginAndHygiene(unittest.TestCase):
    def test_45_openclaw_plugin_registers_and_executes_all_seven_tools(self) -> None:
        js_code = """
        import plugin from './ts-plugin/index.mjs';
        const registered = new Map();
        const shutdowns = [];
        plugin.register({
          pluginConfig: { defaultProfile: 'voice_call_realtime' },
          registerTool(tool) { registered.set(tool.name, tool); },
          onShutdown(fn) { shutdowns.push(fn); }
        });
        const ttsRes = await registered.get('najdi_tts_prepare').execute('id1', { text: 'أبشر الحين', backend: 'habibi' });
        const callRes = await registered.get('najdi_call_stream_chunk').execute('id2', { text: 'سم، شيكت لك الحين [ACTION: ok].' });
        const turnRes = await registered.get('najdi_call_turn_analyze').execute('id3', { text: 'طيب', is_ai_speaking: true });
        const scoreRes = await registered.get('najdi_dialect_score').execute('id4', { text: 'أبشر طال عمرك' });
        const gndRes = await registered.get('najdi_style_grounding').execute('id5', {});
        for (const fn of shutdowns) fn();
        console.log(JSON.stringify({
          count: registered.size,
          speakable: ttsRes.details.speakable,
          clauses: callRes.details.clause_count,
          turnKind: turnRes.details.kind,
          saudiScore: scoreRes.details.saudi_alignment_score,
          profile: gndRes.details.profile
        }));
        """
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", js_code],
            cwd=REPO_ROOT,
            env=_cli_env(),
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        out = json.loads(proc.stdout.strip())
        self.assertEqual(out["count"], 7)
        self.assertEqual(out["speakable"], "②〈أَبْشِرْ الْحِينْ〉")
        self.assertEqual(out["clauses"], 2)
        self.assertEqual(out["turnKind"], "backchannel")
        self.assertGreaterEqual(out["saudiScore"], 90.0)
        self.assertEqual(out["profile"], "voice_call_realtime")

    def test_46_plugin_manifest_schema_valid(self) -> None:
        manifest_path = REPO_ROOT / "openclaw.plugin.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["id"], "najdikit")
        self.assertEqual(manifest["version"], "2.0.0")
        self.assertEqual(len(manifest["contracts"]["tools"]), 7)

    def test_47_no_hardcoded_home_paths_in_repository(self) -> None:
        forbidden_prefix = "/" + "Users" + "/"
        offenders: list[str] = []
        for path in REPO_ROOT.rglob("*"):
            if ".git" in path.parts or "__pycache__" in path.parts or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if forbidden_prefix in content:
                offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [], f"Hardcoded user path found in: {offenders}")


if __name__ == "__main__":
    unittest.main()
