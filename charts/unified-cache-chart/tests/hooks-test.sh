#!/bin/bash
# 生命周期钩子渲染 + 行为回归测试（plan/vllm-lifecycle-hooks-2026-07-10.md §5）。
# 依赖：helm、python3(+PyYAML)、bash。用法：bash tests/hooks-test.sh
# 覆盖：hooks 全量渲染断言 / PD per-role 覆盖与 null 禁用 / 空白串=未配置 /
#       三类渲染期报错 / 双机 entry-worker 分流 / bash -n / preStart export 行为链路。
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH

CHART_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$CHART_DIR"

cat > "$WORK/hooks-full.yaml" <<'EOF'
servingEngineSpec:
  modelSpec:
    terminationGracePeriodSeconds: 120
    hooks:
      preStart: |
        echo "preStart running, MASTER_IP=${MASTER_IP}"
        export UC_HOOK_MARK=1
      postReady: |
        curl -m 5 -s -X POST "http://lb.example/register?ip=${POD_IP}" || true
      preStop: |
        curl -m 5 -s -X POST "http://lb.example/deregister?ip=${POD_IP}" || true
        sleep 5
EOF
cat > "$WORK/hooks-blank.yaml" <<'EOF'
servingEngineSpec:
  modelSpec:
    hooks:
      preStart: |

EOF
cat > "$WORK/hooks-unknown-key.yaml" <<'EOF'
servingEngineSpec:
  modelSpec:
    hooks:
      postStop: |
        echo unknown hook name
EOF
cat > "$WORK/hooks-onexit.yaml" <<'EOF'
servingEngineSpec:
  modelSpec:
    hooks:
      onExit: |
        echo crash scene
EOF
cat > "$WORK/hooks-nonstring.yaml" <<'EOF'
servingEngineSpec:
  modelSpec:
    hooks:
      preStart:
        script: nested-object-not-allowed
EOF

# PD per-role 覆盖 + null 禁用（helm 对 list 整体替换，需整份改写 roles）
python3 - "$WORK/pd-role-hooks.yaml" <<'PYEOF'
import sys, yaml
d = yaml.safe_load(open("models/ascend/values-qwen3-0p6b-1p1-1d1.yaml"))
ms = d["servingEngineSpec"]["modelSpec"]
ms["hooks"] = {"preStop": 'echo "global preStop"\n', "preStart": 'echo "global preStart"\n'}
for r in ms["roles"]:
    if r["name"] == "prefill":
        r["hooks"] = {"preStop": 'echo "prefill preStop override"\n'}
    if r["name"] == "decode":
        r["hooks"] = {"preStop": None, "preStart": None}
yaml.safe_dump(d, open(sys.argv[1], "w"), allow_unicode=True, sort_keys=False)
PYEOF

echo "== 1) hooks 全量（单机 engine）+ bash -n =="
helm template rel . -f values.yaml -f models/ascend/values-qwen3-0p6b-1e1.yaml -f "$WORK/hooks-full.yaml" > "$WORK/render-full.yaml"
python3 - "$WORK/render-full.yaml" "$WORK/hook-pre-start.sh" <<'PYEOF'
import sys, yaml, subprocess, tempfile, os
docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
cms = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "ConfigMap"}
ms  = [d for d in docs if d.get("kind") == "ModelServing"][0]
data = cms["rel-engine-hooks"]["data"]
assert set(data) == {"hook-pre-start.sh", "hook-post-ready.sh", "hook-pre-stop.sh"}
code = [l.strip() for l in data["hook-pre-start.sh"].splitlines() if l.strip() and not l.strip().startswith("#")]
assert not any(l.startswith("set ") or l.startswith("exec ") for l in code), "preStart 包壳泄漏 set/exec"
assert any(l.strip() == "set -e" for l in data["hook-post-ready.sh"].splitlines())
assert "exec >>/proc/1/fd/1 2>&1" in data["hook-pre-stop.sh"]
spec = ms["spec"]["template"]["roles"][0]["entryTemplate"]["spec"]
c = spec["containers"][0]
env = {e["name"]: e.get("value") for e in c["env"]}
assert spec["terminationGracePeriodSeconds"] == 120
assert env.get("UC_POD_KIND") == "entry"
assert c["lifecycle"]["preStop"]["exec"]["command"] == ["/bin/bash", "/vllm-workspace/UnifiedCache/entrypoint/hook-pre-stop.sh"]
assert "rel-engine-hooks" in [s["configMap"]["name"] for s in spec["volumes"][0]["projected"]["sources"]]
entry = cms["rel-vllm-args"]["data"]["args-entrypoint.sh"]
assert entry.index("prepare_common_runtime") < entry.index("hook-pre-start.sh") < entry.index("cmd=(")
assert entry.index("print_server_args") < entry.index("hook-post-ready.sh") < entry.index('exec "${cmd[@]}"')
for name, content in [("args-entrypoint.sh", entry)] + list(data.items()):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(content); p = f.name
    r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n {name}: {r.stderr}"
    os.unlink(p)
open(sys.argv[2], "w").write(data["hook-pre-start.sh"])
print("   PASS")
PYEOF

echo "== 2) PD per-role 覆盖 + decode 全 null 禁用 =="
helm template rel . -f "$WORK/pd-role-hooks.yaml" > "$WORK/render-pd.yaml"
python3 - "$WORK/render-pd.yaml" <<'PYEOF'
import sys, yaml
docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
cms = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "ConfigMap"}
ms  = [d for d in docs if d.get("kind") == "ModelServing"][0]
assert "prefill preStop override" in cms["rel-prefill-hooks"]["data"]["hook-pre-stop.sh"]
assert "global preStart" in cms["rel-prefill-hooks"]["data"]["hook-pre-start.sh"]
assert "rel-decode-hooks" not in cms
roles = {r["name"]: r for r in ms["spec"]["template"]["roles"]}
pc = roles["prefill"]["entryTemplate"]["spec"]["containers"][0]
dc = roles["decode"]["entryTemplate"]["spec"]["containers"][0]
assert "lifecycle" in pc and "lifecycle" not in dc
dsrcs = [s["configMap"]["name"] for s in roles["decode"]["entryTemplate"]["spec"]["volumes"][0]["projected"]["sources"]]
assert not any("hooks" in s for s in dsrcs)
print("   PASS")
PYEOF

echo "== 3) 空白串 = 未配置（覆盖掉模型自带 preStart 后，CM 只剩另两键）=="
helm template rel . -f values.yaml -f models/ascend/values-qwen3-0p6b-1e1.yaml -f "$WORK/hooks-blank.yaml" > "$WORK/render-blank.yaml"
python3 - "$WORK/render-blank.yaml" <<'PYEOF'
import sys, yaml
docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
cms = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "ConfigMap"}
data = cms["rel-engine-hooks"]["data"]
assert "hook-pre-start.sh" not in data, "空白串应视为未配置"
assert set(data) == {"hook-post-ready.sh", "hook-pre-stop.sh"}, data.keys()
entry = cms["rel-vllm-args"]["data"]["args-entrypoint.sh"]
assert "hook-pre-start.sh" in entry, "仍有其它钩子, entrypoint gate 应保留"
print("   PASS")
PYEOF

echo "== 4) 渲染期报错三例 =="
for c in hooks-unknown-key hooks-onexit hooks-nonstring; do
  if helm template rel . -f values.yaml -f models/ascend/values-qwen3-0p6b-1e1.yaml -f "$WORK/$c.yaml" >/dev/null 2>&1; then
    echo "   FAIL: $c 应报错却渲染成功"; exit 1
  fi
done
echo "   PASS"

echo "== 5) 双机 entry/worker 分流 =="
helm template rel . -f values.yaml -f models/ascend/values-qwen3-235b-multi.yaml -f "$WORK/hooks-full.yaml" > "$WORK/render-multi.yaml"
python3 - "$WORK/render-multi.yaml" <<'PYEOF'
import sys, yaml
docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
role = [d for d in docs if d.get("kind") == "ModelServing"][0]["spec"]["template"]["roles"][0]
for kind, tpl in [("entry", "entryTemplate"), ("worker", "workerTemplate")]:
    c = role[tpl]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in c["env"]}
    assert env.get("UC_POD_KIND") == kind and "lifecycle" in c
    if kind == "worker":
        assert "livenessProbe" not in c and "startupProbe" not in c
print("   PASS")
PYEOF

echo "== 6) preStart export 行为链路（渲染产物 source + exec）=="
cd "$WORK"
cat > sim-entry.sh <<'EOF'
#!/bin/bash
set -e
ENTRYPOINT_DIR="."
export MASTER_IP=10.0.0.1
if [[ -f "${ENTRYPOINT_DIR}/hook-pre-start.sh" ]]; then
  source "${ENTRYPOINT_DIR}/hook-pre-start.sh"
fi
exec env
EOF
chmod +x sim-entry.sh
./sim-entry.sh | grep -q "^UC_HOOK_MARK=1$" || { echo "   FAIL: export 未穿透 exec"; exit 1; }
echo "   PASS"

echo "ALL HOOKS TESTS PASSED"
