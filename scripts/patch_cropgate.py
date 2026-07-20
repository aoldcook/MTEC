#!/usr/bin/env python
"""Insert an env-var override so MTEC_DETAIL_MAX_CROPS forces the number of
detail crops regardless of the auto-selected anchor policy (clean single-variable
ablation for Config H). Mirrors the MTEC_DISABLE_TASK_FAMILY_ROUTING gate."""
import io

P = "/root/autodl-tmp/MTEC/scripts/run_modelscope_mtec_anchor_api_full.py"
src = open(P).read()

anchor = ('            selected_policy["detail_max_crops"] = int(selected_policy.get("detail_max_crops") or 0)'
          ' + int(video_query_detail_extra_crops)\n')
if anchor not in src:
    raise SystemExit("PATCH_ANCHOR_NOT_FOUND")

gate = anchor + (
    '        _force_crops = os.environ.get("MTEC_DETAIL_MAX_CROPS")\n'
    '        if _force_crops is not None:\n'
    '            selected_policy = dict(selected_policy)\n'
    '            selected_policy["detail_max_crops"] = int(_force_crops)\n'
)

if 'MTEC_DETAIL_MAX_CROPS' in src:
    print("ALREADY_PATCHED")
else:
    src = src.replace(anchor, gate, 1)
    open(P, "w").write(src)
    print("PATCHED")

# verify os is imported
import ast
ast.parse(open(P).read())
print("SYNTAX_OK; os_imported=", "import os" in open(P).read().split("def ")[0])
