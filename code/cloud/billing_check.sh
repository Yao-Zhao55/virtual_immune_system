#!/usr/bin/env bash
# Ask the ECS API how this instance is billed, using the instance RAM role.
set -u
META=http://100.100.100.200/latest/meta-data/ram/security-credentials/ECSOssRole
JSON=$(curl -s -m 5 "$META")
AK=$(printf '%s' "$JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)["AccessKeyId"])')
SK=$(printf '%s' "$JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)["AccessKeySecret"])')
TK=$(printf '%s' "$JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)["SecurityToken"])')
echo "== instance billing =="
aliyun --access-key-id "$AK" --access-key-secret "$SK" --sts-token "$TK" --region cn-shenzhen \
  ecs DescribeInstances --InstanceIds '["i-wz9453n2tirhiifus3o5"]' 2>&1 \
  | python3 -c '
import json,sys
raw=sys.stdin.read()
try:
    data=json.loads(raw)
except Exception:
    print(raw[:800]); raise SystemExit
for inst in data.get("Instances",{}).get("Instance",[]):
    keep={k:inst.get(k) for k in ("InstanceId","InstanceChargeType","SpotStrategy","InstanceType","ExpiredTime","CreationTime","Status","ZoneId")}
    print(json.dumps(keep,ensure_ascii=False,indent=2))
'
