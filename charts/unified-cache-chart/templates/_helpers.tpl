{{/*
Define ports for the pods
*/}}
{{- define "chart.container-port" -}}
{{-  default "8000" .Values.servingEngineSpec.containerPort }}
{{- end }}

{{/*
Define service account name for engine pods.
*/}}
{{- define "chart.engineServiceAccountName" -}}
{{- $modelSpec := .Values.servingEngineSpec.modelSpec -}}
{{- if and $modelSpec (hasKey $modelSpec "serviceAccountName") $modelSpec.serviceAccountName -}}
{{- $modelSpec.serviceAccountName -}}
{{- else -}}
{{- printf "%s-engine-service-account" .Release.Name -}}
{{- end -}}
{{- end }}

{{/*
Normalize a value so it can be used as part of a Kubernetes DNS label.
*/}}
{{- define "chart.k8sNamePart" -}}
{{- $raw := . | toString | lower | replace "_" "-" -}}
{{- $clean := regexReplaceAll "[^a-z0-9-]" $raw "-" | trimAll "-" -}}
{{- if eq $clean "" -}}x{{- else -}}{{- $clean -}}{{- end -}}
{{- end -}}

{{/*
Keep a generated Kubernetes name within 63 characters with a stable hash suffix.
*/}}
{{- define "chart.truncatedK8sName" -}}
{{- $raw := . | toString | lower | replace "_" "-" -}}
{{- $clean := regexReplaceAll "[^a-z0-9-]" $raw "-" | trimAll "-" -}}
{{- if eq $clean "" -}}
x
{{- else if le (len $clean) 63 -}}
{{- $clean -}}
{{- else -}}
{{- $hash := sha256sum $clean | trunc 10 -}}
{{- $prefixLen := sub 62 (len $hash) | int -}}
{{- printf "%s-%s" ($clean | trunc $prefixLen | trimSuffix "-") $hash -}}
{{- end -}}
{{- end -}}

{{/*
Build a per-release cluster-scoped resource name. Cluster-scoped resources
cannot have metadata.namespace, so namespace is encoded in the object name.
*/}}
{{- define "chart.clusterScopedInstanceName" -}}
{{- $root := index . 0 -}}
{{- $purpose := index . 1 -}}
{{- $base := printf "%s-%s-%s" (include "chart.k8sNamePart" $root.Release.Namespace) (include "chart.k8sNamePart" $root.Release.Name) (include "chart.k8sNamePart" $purpose) -}}
{{- include "chart.truncatedK8sName" $base -}}
{{- end -}}

{{- define "chart.enginePvReaderClusterRoleName" -}}
{{- include "chart.clusterScopedInstanceName" (list . "engine-pv-reader") -}}
{{- end -}}

{{/*
Define container port name
*/}}
{{- define "chart.container-port-name" -}}
"container-port"
{{- end }}

{{/*
Extract the tensor-parallel size from the vllmArgs block.
Supports --tensor-parallel-size / -tp, space- or =-separated. The block is
line-based but a flag and its value may share a line, so each line is
whitespace-normalised and scanned token-by-token. Returns "" when unset
(TP=1, no /dev/shm needed).
*/}}
{{- define "chart.vllmTpSize" -}}
{{- $args := default "" (index . 0) -}}
{{- $result := "" -}}
{{- range $line := splitList "\n" $args -}}
{{- $trimmed := trim $line -}}
{{- if and $trimmed (not (hasPrefix "#" $trimmed)) -}}
{{- $fields := splitList " " (trim (regexReplaceAll "[ \t]+" $trimmed " ")) -}}
{{- range $i, $f := $fields -}}
{{- if and (or (eq $f "--tensor-parallel-size") (eq $f "-tp")) (lt (add $i 1) (len $fields)) -}}
{{- $result = trim (index $fields (add $i 1)) -}}
{{- else if hasPrefix "--tensor-parallel-size=" $f -}}
{{- $result = trim (trimPrefix "--tensor-parallel-size=" $f) -}}
{{- else if hasPrefix "-tp=" $f -}}
{{- $result = trim (trimPrefix "-tp=" $f) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- $result -}}
{{- end -}}


{{/*
Return modelSpec.storage.extraStorage. (chipType / chipExtraStorage 已删除)
Usage: include "chart.combinedExtraStorage" (list $modelSpec $)
*/}}
{{- define "chart.combinedExtraStorage" -}}
{{- $modelSpec := index . 0 -}}
{{- $storage := default (dict) $modelSpec.storage -}}
{{- toYaml (default (list) $storage.extraStorage) -}}
{{- end -}}

{{/*
为镜像应用全局 registry（若已包含 registry 则不重复添加）
Usage: include "chart.resolveImageWithRegistry" (list $image $)
*/}}
{{- define "chart.resolveImageWithRegistry" -}}
{{- $image := default "" (index . 0) -}}
{{- $global := index . 1 -}}
{{- $registry := $global.Values.images.registry | default "" -}}
{{- if and $registry $image -}}
  {{- if contains "/" $image -}}
    {{- $first := index (splitList "/" $image) 0 -}}
    {{- if or (contains "." $first) (contains ":" $first) (eq $first "localhost") -}}
{{- $image -}}
    {{- else -}}
{{- printf "%s/%s" $registry $image -}}
    {{- end -}}
  {{- else -}}
{{- printf "%s/%s" $registry $image -}}
  {{- end -}}
{{- else -}}
{{- $image -}}
{{- end -}}
{{- end -}}

{{/*
Resolve a release image reference. A structured repository/tag/digest object
takes precedence over the legacy string. When both digest and tag are set,
the immutable digest is rendered. values.schema.json rejects orphan fields;
the template repeats those guards so rendering is safe without schema support.
Usage: include "chart.resolveReleaseImage" (list $structured $legacy $root)
*/}}
{{- define "chart.resolveReleaseImage" -}}
{{- $structured := default (dict) (index . 0) -}}
{{- $legacy := default "" (index . 1) -}}
{{- $root := index . 2 -}}
{{- $repository := default "" $structured.repository -}}
{{- $tag := default "" $structured.tag -}}
{{- $digest := default "" $structured.digest -}}
{{- $image := "" -}}
{{- if $repository -}}
  {{- if $digest -}}
    {{- $image = printf "%s@%s" $repository $digest -}}
  {{- else if $tag -}}
    {{- $image = printf "%s:%s" $repository $tag -}}
  {{- else -}}
    {{- fail "structured image repository requires tag or digest" -}}
  {{- end -}}
{{- else -}}
  {{- if or $tag $digest -}}
    {{- fail "structured image tag/digest requires repository" -}}
  {{- end -}}
  {{- $image = $legacy -}}
{{- end -}}
{{- if empty $image -}}
  {{- fail "image reference is empty" -}}
{{- end -}}
{{- include "chart.resolveImageWithRegistry" (list $image $root) -}}
{{- end -}}

{{- define "chart.releaseEngineImage" -}}
{{- $root := . -}}
{{- $images := default (dict) $root.Values.images -}}
{{- include "chart.resolveReleaseImage" (list (default (dict) $images.engine) $images.image $root) -}}
{{- end -}}

{{- define "chart.releaseMooncakeMasterImage" -}}
{{- $root := . -}}
{{- $images := default (dict) $root.Values.images -}}
{{- $structured := default (dict) $images.mooncakeMaster -}}
{{- $legacy := default "" $images.mooncakeMasterImage -}}
{{- if and (empty $structured) (empty $legacy) -}}
{{- include "chart.releaseEngineImage" $root -}}
{{- else -}}
{{- include "chart.resolveReleaseImage" (list $structured $legacy $root) -}}
{{- end -}}
{{- end -}}

{{/* ===================== Mooncake master / client helpers ===================== */}}

{{- define "chart.mooncakeMasterName" -}}
{{- $mm := default (dict) .Values.mooncakeMaster -}}
{{- if $mm.nameOverride -}}
{{- include "chart.truncatedK8sName" $mm.nameOverride -}}
{{- else -}}
{{- include "chart.truncatedK8sName" (printf "%s-mooncake-master" .Release.Name) -}}
{{- end -}}
{{- end -}}

{{- define "chart.mooncakeClientConfigName" -}}
{{- include "chart.truncatedK8sName" (printf "%s-mooncake-client-config" .Release.Name) -}}
{{- end -}}

{{- define "chart.mooncakeConfigMountDir" -}}
/vllm-workspace/UnifiedCache/mooncake
{{- end -}}

{{- define "chart.mooncakeConfigPath" -}}
{{ include "chart.mooncakeConfigMountDir" . }}/mooncake.json
{{- end -}}

{{- define "chart.pdMooncakeMasterRequested" -}}
{{- $modelSpec := index . 0 -}}
{{- $pd := default (dict) $modelSpec.pd -}}
{{- $mooncake := default (dict) $pd.mooncake -}}
{{- $master := default (dict) $mooncake.master -}}
{{- if default false $master.enabled -}}true{{- else -}}false{{- end -}}
{{- end -}}

{{- define "chart.pdMooncakeMasterEnabled" -}}
{{- $modelSpec := index . 0 -}}
{{- $pd := default (dict) $modelSpec.pd -}}
{{- $kv := default (dict) $pd.kvTransfer -}}
{{- if and (eq (include "chart.pdMooncakeMasterRequested" (list $modelSpec)) "true") (eq (default "" $kv.routerType) "mooncake") -}}true{{- else -}}false{{- end -}}
{{- end -}}

{{- define "chart.mooncakeMasterRpcAddress" -}}
{{- $mm := default (dict) .Values.mooncakeMaster -}}
{{- $service := default (dict) $mm.service -}}
{{- if default false $mm.create -}}
{{- printf "%s.%s.svc:%v" (include "chart.mooncakeMasterName" .) .Release.Namespace (default 50088 $service.rpcPort) -}}
{{- else -}}
{{- default "" (default (dict) $mm.external).rpcAddress -}}
{{- end -}}
{{- end -}}

{{- define "chart.mooncakeClientConfigUser" -}}
{{- $mm := default (dict) .Values.mooncakeMaster -}}
{{- $client := default (dict) $mm.client -}}
{{- $raw := default (dict) $client.config -}}
{{- $cfg := dict -}}
{{- if kindIs "string" $raw -}}
{{- $trimmed := trim $raw -}}
{{- if $trimmed -}}
{{- $parsed := fromJson $trimmed -}}
{{- if not (kindIs "map" $parsed) -}}
{{- fail "mooncakeMaster.client.config must be a JSON object when set as a string block" -}}
{{- end -}}
{{- $cfg = $parsed -}}
{{- end -}}
{{- else if kindIs "map" $raw -}}
{{- $cfg = $raw -}}
{{- else -}}
{{- fail "mooncakeMaster.client.config must be either a map or a JSON object string" -}}
{{- end -}}
{{- toYaml $cfg -}}
{{- end -}}

{{- define "chart.mooncakeClientConfigResolved" -}}
{{- $defaults := dict "metadata_server" "P2PHANDSHAKE" "device_name" "" "global_segment_size" "1GB" "preferred_segment" false "prefer_alloc_in_same_node" true -}}
{{- $cfg := fromYaml (toYaml $defaults) -}}
{{- $userCfg := include "chart.mooncakeClientConfigUser" . | fromYaml -}}
{{- $cfg = mergeOverwrite $cfg (default (dict) $userCfg) -}}
{{- toYaml $cfg -}}
{{- end -}}

{{- define "chart.mooncakeGlobalSegmentSize" -}}
{{- $cfg := include "chart.mooncakeClientConfigResolved" . | fromYaml -}}
{{- default "" (get $cfg "global_segment_size") -}}
{{- end -}}

{{- define "chart.validateNoManagedMooncakeEnv" -}}
{{- $envs := default (list) (index . 0) -}}
{{- $where := index . 1 -}}
{{- $managed := list "MOONCAKE_CONFIG_PATH" "MOONCAKE_MASTER" "MOONCAKE_GLOBAL_SEGMENT_SIZE" -}}
{{- range $idx, $env := $envs -}}
{{- if and (kindIs "map" $env) (has (default "" $env.name) $managed) -}}
{{- fail (printf "%s[%d].name=%s is chart-managed when pd.mooncake.master.enabled=true" $where $idx $env.name) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "chart.validateMooncakeMaster" -}}
{{- $modelSpec := index . 0 -}}
{{- $root := index . 1 -}}
{{- $pd := default (dict) $modelSpec.pd -}}
{{- if eq (include "chart.pdMooncakeMasterRequested" (list $modelSpec)) "true" -}}
{{- $kv := default (dict) $pd.kvTransfer -}}
{{- if ne (default "" $kv.routerType) "mooncake" -}}
{{- fail "servingEngineSpec.modelSpec.pd.mooncake.master.enabled=true requires servingEngineSpec.modelSpec.pd.kvTransfer.routerType=mooncake" -}}
{{- end -}}
{{- $mm := default (dict) $root.Values.mooncakeMaster -}}
{{- if not (default false $mm.enabled) -}}
{{- fail "servingEngineSpec.modelSpec.pd.mooncake.master.enabled=true requires top-level mooncakeMaster.enabled=true" -}}
{{- end -}}
{{- if and (not (default false $mm.create)) (empty (default "" (default (dict) $mm.external).rpcAddress)) -}}
{{- fail "mooncakeMaster.external.rpcAddress is required when mooncakeMaster.enabled=true and mooncakeMaster.create=false" -}}
{{- end -}}
{{- $client := default (dict) $mm.client -}}
{{- $cfg := include "chart.mooncakeClientConfigUser" $root | fromYaml -}}
{{- if hasKey $cfg "master_server_address" -}}
{{- fail "mooncakeMaster.client.config.master_server_address is chart-managed; set mooncakeMaster.external.rpcAddress or mooncakeMaster.create instead" -}}
{{- end -}}
{{- $resolvedCfg := include "chart.mooncakeClientConfigResolved" $root | fromYaml -}}
{{- if not (get $resolvedCfg "protocol") -}}
{{- fail "mooncakeMaster.client.config.protocol is required when Mooncake master is enabled; set it in the model values, e.g. rdma for CUDA/GPU or ascend for Ascend/NPU" -}}
{{- end -}}
{{- include "chart.validateNoManagedMooncakeEnv" (list $client.env "mooncakeMaster.client.env") -}}
{{- include "chart.validateNoManagedMooncakeEnv" (list $modelSpec.env "servingEngineSpec.modelSpec.env") -}}
{{- end -}}
{{- end -}}

{{/*
Get storage backend volume name from item name.
Usage: include "chart.storageVolumeName" (list $ $item)
*/}}
{{- define "chart.storageVolumeName" -}}
{{- $root := index . 0 -}}
{{- $item := index . 1 -}}
{{- printf "%s-storage-%s" $root.Release.Name ($item.name | toString | lower | replace "_" "-") | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Get dynamic PVC claim name from storage item.
Usage: include "chart.storageDynamicPvcClaimName" (list $ $modelSpec $item)
*/}}
{{- define "chart.storageDynamicPvcClaimName" -}}
{{- $root := index . 0 -}}
{{- $modelSpec := index . 1 -}}
{{- $item := index . 2 -}}
{{- printf "%s-%s-%s-storage-claim" $root.Release.Name ($modelSpec.name | toString | lower | replace "_" "-") ($item.name | toString | lower | replace "_" "-") | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Get static PV name from storage item.
Usage: include "chart.storageStaticPvName" (list $ $modelSpec $item)
*/}}
{{- define "chart.storageStaticPvName" -}}
{{- $root := index . 0 -}}
{{- $modelSpec := index . 1 -}}
{{- $item := index . 2 -}}
{{- $base := printf "%s-%s-%s-%s-static-pv" (include "chart.k8sNamePart" $root.Release.Namespace) (include "chart.k8sNamePart" $root.Release.Name) (include "chart.k8sNamePart" $modelSpec.name) (include "chart.k8sNamePart" $item.name) -}}
{{- include "chart.truncatedK8sName" $base -}}
{{- end -}}

{{/*
Get static PVC claim name from storage item.
Usage: include "chart.storageStaticPvcClaimName" (list $ $modelSpec $item)
*/}}
{{- define "chart.storageStaticPvcClaimName" -}}
{{- $root := index . 0 -}}
{{- $modelSpec := index . 1 -}}
{{- $item := index . 2 -}}
{{- printf "%s-%s-%s-static-pvc" $root.Release.Name ($modelSpec.name | toString | lower | replace "_" "-") ($item.name | toString | lower | replace "_" "-") | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Validate one storage item. Fail on invalid schema.
Usage: include "chart.validateStorageItem" (list $item $idx $path)
*/}}
{{- define "chart.validateStorageItem" -}}
{{- $item := index . 0 -}}
{{- $idx := index . 1 -}}
{{- $path := index . 2 -}}
{{- if not (kindIs "map" $item) -}}
{{- fail (printf "%s[%d] must be an object" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item "name")) (empty $item.name) -}}
{{- fail (printf "%s[%d].name is required" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item "mountPath")) (empty $item.mountPath) -}}
{{- fail (printf "%s[%d].mountPath is required" $path $idx) -}}
{{- end -}}
{{- $sourceCount := 0 -}}
{{- if hasKey $item "dynamicPVC" -}}
{{- $sourceCount = add $sourceCount 1 -}}
{{- end -}}
{{- if hasKey $item "staticPVC" -}}
{{- $sourceCount = add $sourceCount 1 -}}
{{- end -}}
{{- if hasKey $item "persistentVolumeClaim" -}}
{{- $sourceCount = add $sourceCount 1 -}}
{{- end -}}
{{- if hasKey $item "hostPath" -}}
{{- $sourceCount = add $sourceCount 1 -}}
{{- end -}}
{{- if hasKey $item "csi" -}}
{{- $sourceCount = add $sourceCount 1 -}}
{{- end -}}
{{- if hasKey $item "nfs" -}}
{{- $sourceCount = add $sourceCount 1 -}}
{{- end -}}
{{- if ne $sourceCount 1 -}}
{{- fail (printf "%s[%d] must have exactly one source: dynamicPVC/staticPVC/persistentVolumeClaim/hostPath/csi/nfs" $path $idx) -}}
{{- end -}}
{{- if hasKey $item "dynamicPVC" -}}
{{- if not (kindIs "map" $item.dynamicPVC) -}}
{{- fail (printf "%s[%d].dynamicPVC must be an object" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item.dynamicPVC "pvcStorage")) (empty $item.dynamicPVC.pvcStorage) -}}
{{- fail (printf "%s[%d].dynamicPVC.pvcStorage is required" $path $idx) -}}
{{- end -}}
{{- end -}}
{{- if hasKey $item "staticPVC" -}}
{{- if not (kindIs "map" $item.staticPVC) -}}
{{- fail (printf "%s[%d].staticPVC must be an object" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item.staticPVC "pvcStorage")) (empty $item.staticPVC.pvcStorage) -}}
{{- fail (printf "%s[%d].staticPVC.pvcStorage is required" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item.staticPVC "csi")) (empty $item.staticPVC.csi) -}}
{{- fail (printf "%s[%d].staticPVC.csi is required" $path $idx) -}}
{{- end -}}
{{- if not (kindIs "map" $item.staticPVC.csi) -}}
{{- fail (printf "%s[%d].staticPVC.csi must be an object" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item.staticPVC.csi "driver")) (empty $item.staticPVC.csi.driver) -}}
{{- fail (printf "%s[%d].staticPVC.csi.driver is required" $path $idx) -}}
{{- end -}}
{{- end -}}
{{- if hasKey $item "persistentVolumeClaim" -}}
{{- if not (kindIs "map" $item.persistentVolumeClaim) -}}
{{- fail (printf "%s[%d].persistentVolumeClaim must be an object" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item.persistentVolumeClaim "claimName")) (empty $item.persistentVolumeClaim.claimName) -}}
{{- fail (printf "%s[%d].persistentVolumeClaim.claimName is required" $path $idx) -}}
{{- end -}}
{{- end -}}
{{- if hasKey $item "nfs" -}}
{{- if not (kindIs "map" $item.nfs) -}}
{{- fail (printf "%s[%d].nfs must be an object" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item.nfs "server")) (empty $item.nfs.server) -}}
{{- fail (printf "%s[%d].nfs.server is required" $path $idx) -}}
{{- end -}}
{{- if or (not (hasKey $item.nfs "path")) (empty $item.nfs.path) -}}
{{- fail (printf "%s[%d].nfs.path is required" $path $idx) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Validate storage schema and uniqueness.
Usage: include "chart.validateStorageConfig" (list $modelSpec $)
*/}}
{{- define "chart.validateStorageConfig" -}}
{{- $modelSpec := index . 0 -}}
{{- $global := dict -}}
{{- if gt (len .) 1 -}}
{{- $global = index . 1 -}}
{{- end -}}
{{- if and (hasKey $modelSpec "storage") (not (kindIs "map" $modelSpec.storage)) -}}
{{- fail "servingEngineSpec.modelSpec.storage must be an object" -}}
{{- end -}}
{{- $storage := default (dict) $modelSpec.storage -}}
{{- $unifiedStorage := default (list) $storage.unifiedcacheStorage -}}
{{- $extraStorage := default (list) $storage.extraStorage -}}
{{- if not (kindIs "slice" $unifiedStorage) -}}
{{- fail "servingEngineSpec.modelSpec.storage.unifiedcacheStorage must be a list" -}}
{{- end -}}
{{- if not (kindIs "slice" $extraStorage) -}}
{{- fail "servingEngineSpec.modelSpec.storage.extraStorage must be a list" -}}
{{- end -}}
{{- $seenNames := dict -}}
{{- range $idx, $item := $unifiedStorage -}}
{{- include "chart.validateStorageItem" (list $item $idx "servingEngineSpec.modelSpec.storage.unifiedcacheStorage") -}}
{{- $name := $item.name | toString -}}
{{- $normalizedName := $name | lower | replace "_" "-" -}}
{{- if hasKey $seenNames $normalizedName -}}
{{- fail (printf "duplicate storage name %q in servingEngineSpec.modelSpec.storage" $name) -}}
{{- end -}}
{{- $_ := set $seenNames $normalizedName true -}}
{{- end -}}
{{- range $idx, $item := $extraStorage -}}
{{- include "chart.validateStorageItem" (list $item $idx "servingEngineSpec.modelSpec.storage.extraStorage") -}}
{{- $name := $item.name | toString -}}
{{- $normalizedName := $name | lower | replace "_" "-" -}}
{{- if hasKey $seenNames $normalizedName -}}
{{- fail (printf "duplicate storage name %q in servingEngineSpec.modelSpec.storage" $name) -}}
{{- end -}}
{{- $_ := set $seenNames $normalizedName true -}}
{{- end -}}
{{- end -}}

{{/* Validate UCM container/switch/config types once for every consumer helper. */}}
{{- define "chart.validateUcmConfigShape" -}}
{{- $modelSpec := index . 0 -}}
{{- if hasKey $modelSpec "unifiedcacheConfig" -}}
{{- if not (kindIs "map" $modelSpec.unifiedcacheConfig) -}}
{{- fail "servingEngineSpec.modelSpec.unifiedcacheConfig must be a map" -}}
{{- end -}}
{{- $ucm := $modelSpec.unifiedcacheConfig -}}
{{- range $switchName := list "enabled" "enable" -}}
{{- if and (hasKey $ucm $switchName) (not (kindIs "bool" (get $ucm $switchName))) -}}
{{- fail (printf "servingEngineSpec.modelSpec.unifiedcacheConfig.%s must be a boolean" $switchName) -}}
{{- end -}}
{{- end -}}
{{- if and (hasKey $ucm "enabled") (hasKey $ucm "enable") (ne $ucm.enabled $ucm.enable) -}}
{{- fail (printf "servingEngineSpec.modelSpec.unifiedcacheConfig has conflicting switches: enabled=%v vs enable=%v; keep only one (enabled is canonical)" $ucm.enabled $ucm.enable) -}}
{{- end -}}
{{- if and (hasKey $ucm "config") (not (kindIs "map" $ucm.config)) -}}
{{- fail "servingEngineSpec.modelSpec.unifiedcacheConfig.config must be a map" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Returns "true" when UCM is in use: unifiedcacheConfig.config present and non-empty,
AND the explicit switch unifiedcacheConfig.enabled (alias: enable; default true) is not false.
*/}}
{{- define "chart.ucmConfigEnabled" -}}
{{- $modelSpec := index . 0 -}}
{{- include "chart.validateUcmConfigShape" (list $modelSpec) -}}
{{- $ucm := dict -}}
{{- if and (hasKey $modelSpec "unifiedcacheConfig") (kindIs "map" $modelSpec.unifiedcacheConfig) -}}
{{- $ucm = $modelSpec.unifiedcacheConfig -}}
{{- end -}}
{{- $switch := true -}}
{{- if hasKey $ucm "enabled" -}}
{{- $switch = $ucm.enabled -}}
{{- else if hasKey $ucm "enable" -}}
{{- $switch = $ucm.enable -}}
{{- end -}}
{{- if and $switch (hasKey $ucm "config") (kindIs "map" $ucm.config) (gt (len $ucm.config) 0) -}}
true
{{- else -}}
false
{{- end -}}
{{- end -}}

{{/*
Parse the last occurrence of a positive integer vLLM CLI option from a block.
Usage: include "chart.vllmArgInt" (list $args (list "--long-name" "-x") 1)
*/}}
{{- define "chart.vllmArgInt" -}}
{{- $args := default "" (index . 0) -}}
{{- $names := index . 1 -}}
{{- $result := toString (index . 2) -}}
{{- range $line := splitList "\n" $args -}}
{{- $trimmed := trim $line -}}
{{- if and $trimmed (not (hasPrefix "#" $trimmed)) -}}
{{- $fields := splitList " " (trim (regexReplaceAll "[ \t]+" $trimmed " ")) -}}
{{- range $i, $rawField := $fields -}}
{{/* Match the token shape produced by parse-vllm-args.py/shlex, including quoted fragments. */}}
{{- $field := replace "_" "-" (replace "\\" "" (replace "'" "" (replace "\"" "" $rawField))) -}}
{{- if has $field $names -}}
{{- if not (lt (add $i 1) (len $fields)) -}}
{{- fail (printf "vllmArgs option %s is missing its integer value" $field) -}}
{{- end -}}
{{- $rawValue := index $fields (add $i 1) -}}
{{- $result = trim (replace "\\" "" (replace "'" "" (replace "\"" "" $rawValue))) -}}
{{- else -}}
{{- range $name := $names -}}
{{- if hasPrefix (printf "%s=" $name) $field -}}
{{- $result = trim (trimPrefix (printf "%s=" $name) $field) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if not (regexMatch "^[1-9][0-9]*$" $result) -}}
{{- fail (printf "vllmArgs option %s must be a positive integer, got %q" (join "/" $names) $result) -}}
{{- end -}}
{{- $result -}}
{{- end -}}

{{/* Conservative number of ports a logical engine may occupy. */}}
{{- define "chart.kvTransferPortSpan" -}}
{{- $args := default "" (index . 0) -}}
{{- $dp := int (include "chart.vllmArgInt" (list $args (list "--data-parallel-size" "-dp") 1)) -}}
{{- $tp := int (include "chart.vllmArgInt" (list $args (list "--tensor-parallel-size" "-tp") 1)) -}}
{{- $pp := int (include "chart.vllmArgInt" (list $args (list "--pipeline-parallel-size" "-pp") 1)) -}}
{{- $pcp := int (include "chart.vllmArgInt" (list $args (list "--prefill-context-parallel-size" "-pcp") 1)) -}}
{{- $dcp := int (include "chart.vllmArgInt" (list $args (list "--decode-context-parallel-size" "-dcp") 1)) -}}
{{- $cp := int (include "chart.vllmArgInt" (list $args (list "--context-parallel-size") 1)) -}}
{{/* PCP/DCP may describe different phases, but multiplying all configured factors is the safe reservation bound. */}}
{{- mul $dp $tp $pp $pcp $dcp $cp -}}
{{- end -}}

{{/* ===================== PD KV transfer contract ===================== */}}

{{/* Return the exact capability record for a supported vLLM connector. */}}
{{- define "chart.kvTransferCapability" -}}
{{- $connector := default "" (index . 0) -}}
{{- if eq $connector "MooncakeConnectorV1" -}}
{{- toYaml (dict "routerType" "mooncake" "usesPort" true "supportsUcm" true "v1" true) -}}
{{- else if eq $connector "MooncakeHybridConnector" -}}
{{- toYaml (dict "routerType" "mooncake" "usesPort" true "supportsUcm" true "v1" false) -}}
{{- else if eq $connector "NixlConnector" -}}
{{- toYaml (dict "routerType" "nixl" "usesPort" false "supportsUcm" false "v1" false) -}}
{{- else -}}
{}
{{- end -}}
{{- end -}}

{{/* A non-empty pd block is a PD deployment and must use the new kvTransfer schema. */}}
{{- define "chart.pdEnabled" -}}
{{- $modelSpec := index . 0 -}}
{{- $pd := dict -}}
{{- if and (hasKey $modelSpec "pd") (kindIs "map" $modelSpec.pd) -}}
{{- $pd = $modelSpec.pd -}}
{{- end -}}
{{- if gt (len $pd) 0 -}}true{{- else -}}false{{- end -}}
{{- end -}}

{{/*
Single role-level UCM decision used by both KV JSON and Pod env/volumes.
NIXL + effective UCM is rejected by chart.validateKvTransfer before this helper is consumed.
*/}}
{{- define "chart.usesUcmForRole" -}}
{{- $role := index . 0 -}}
{{- $modelSpec := index . 1 -}}
{{- if ne (include "chart.ucmConfigEnabled" (list $modelSpec)) "true" -}}
false
{{- else if eq (include "chart.pdEnabled" (list $modelSpec)) "true" -}}
{{- if eq (include "chart.kvRoleOf" (list $role $modelSpec)) "producer" -}}true{{- else -}}false{{- end -}}
{{- else -}}
true
{{- end -}}
{{- end -}}

{{/* Runtime identity and the role-level UCM decision are chart-owned envs. */}}
{{- define "chart.validateNoManagedKvTransferEnv" -}}
{{- $envs := default (list) (index . 0) -}}
{{- $where := index . 1 -}}
{{- $managed := list "UC_PD_GROUP_NAME" "UC_PD_ROLE_ID" "UC_USES_UCM" "UC_SKIP_KV_CONNECTOR_REGISTRY_PROBE" "VLLM_ARGS_FILE" -}}
{{- range $idx, $env := $envs -}}
{{- if and (kindIs "map" $env) (has (default "" $env.name) $managed) -}}
{{- fail (printf "%s[%d].name=%s is chart-managed by the KV-transfer runtime" $where $idx $env.name) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* nodeTopologyConfig is sourced as shell assignments, so it must not override KV runtime identity/state. */}}
{{- define "chart.validateNoManagedKvTransferTopologyEnv" -}}
{{- $envs := index . 0 -}}
{{- $where := index . 1 -}}
{{- if not (kindIs "map" $envs) -}}
{{- fail (printf "%s must be an environment-variable map" $where) -}}
{{- end -}}
{{- $managed := list "UC_PD_GROUP_NAME" "UC_PD_ROLE_ID" "UC_USES_UCM" "UC_SKIP_KV_CONNECTOR_REGISTRY_PROBE" "VLLM_ARGS_FILE" -}}
{{- range $name, $_ := $envs -}}
{{- if not (regexMatch "^[A-Za-z_][A-Za-z0-9_]*$" $name) -}}
{{- fail (printf "%s contains invalid environment-variable name %q" $where $name) -}}
{{- end -}}
{{- if has $name $managed -}}
{{- fail (printf "%s.%s is chart-managed by the KV-transfer runtime" $where $name) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Validate the public PD/KV transfer interface once, before any role is rendered. */}}
{{- define "chart.validateKvTransfer" -}}
{{- $modelSpec := index . 0 -}}
{{- $root := index . 1 -}}
{{- include "chart.validateUcmConfigShape" (list $modelSpec) -}}
{{- include "chart.validateNoManagedKvTransferEnv" (list $modelSpec.env "servingEngineSpec.modelSpec.env") -}}
{{- $moonMaster := default (dict) $root.Values.mooncakeMaster -}}
{{- $moonClient := default (dict) $moonMaster.client -}}
{{- include "chart.validateNoManagedKvTransferEnv" (list $moonClient.env "mooncakeMaster.client.env") -}}
{{- $topology := default (dict) $root.Values.nodeTopologyConfig -}}
{{- if not (kindIs "map" $topology) -}}
{{- fail "nodeTopologyConfig must be a per-node map" -}}
{{- end -}}
{{- range $node, $envs := $topology -}}
{{- include "chart.validateNoManagedKvTransferTopologyEnv" (list $envs (printf "nodeTopologyConfig.%s" $node)) -}}
{{- end -}}
{{- if and (hasKey $modelSpec "pd") (not (kindIs "map" $modelSpec.pd)) -}}
{{- fail "servingEngineSpec.modelSpec.pd must be a map" -}}
{{- end -}}
{{- $pd := default (dict) $modelSpec.pd -}}
{{- if hasKey $pd "connector" -}}
{{- fail "servingEngineSpec.modelSpec.pd.connector was removed; use pd.kvTransfer.connector and pd.kvTransfer.routerType" -}}
{{- end -}}
{{- if hasKey $pd "mooncakePort" -}}
{{- fail "servingEngineSpec.modelSpec.pd.mooncakePort was removed; use pd.kvTransfer.identity.kvPortBase and instanceStride" -}}
{{- end -}}
{{- if hasKey $pd "ucm" -}}
{{- fail "servingEngineSpec.modelSpec.pd.ucm was removed; use unifiedcacheConfig.enabled as the only UCM switch" -}}
{{- end -}}
{{- if gt (len $pd) 0 -}}
{{- if not (hasKey $pd "kvTransfer") -}}
{{- fail "servingEngineSpec.modelSpec.pd.kvTransfer is required for a PD deployment" -}}
{{- end -}}
{{- if not (kindIs "map" $pd.kvTransfer) -}}
{{- fail "servingEngineSpec.modelSpec.pd.kvTransfer must be a map" -}}
{{- end -}}
{{- $kv := $pd.kvTransfer -}}
{{- $connector := required "servingEngineSpec.modelSpec.pd.kvTransfer.connector is required" $kv.connector -}}
{{- $routerType := required "servingEngineSpec.modelSpec.pd.kvTransfer.routerType is required" $kv.routerType -}}
{{- if or (not (kindIs "string" $connector)) (not (kindIs "string" $routerType)) -}}
{{- fail "servingEngineSpec.modelSpec.pd.kvTransfer.connector and routerType must be strings" -}}
{{- end -}}
{{- $capability := include "chart.kvTransferCapability" (list $connector) | fromYaml -}}
{{- if eq (len $capability) 0 -}}
{{- if has $connector (list "MultiConnector" "UCMConnector") -}}
{{- fail (printf "pd.kvTransfer.connector=%s is chart-managed and cannot be selected directly" $connector) -}}
{{- end -}}
{{- fail (printf "unsupported pd.kvTransfer.connector=%q; supported exact names are MooncakeConnectorV1, MooncakeHybridConnector, NixlConnector" $connector) -}}
{{- end -}}
{{- if ne $routerType (get $capability "routerType") -}}
{{- fail (printf "pd.kvTransfer.connector=%s requires routerType=%s, got %s" $connector (get $capability "routerType") $routerType) -}}
{{- end -}}
{{- $prefill := required "servingEngineSpec.modelSpec.pd.prefill is required" $pd.prefill -}}
{{- $decode := required "servingEngineSpec.modelSpec.pd.decode is required" $pd.decode -}}
{{- if or (not (kindIs "string" $prefill)) (not (kindIs "string" $decode)) -}}
{{- fail "servingEngineSpec.modelSpec.pd.prefill and pd.decode must be role-name strings" -}}
{{- end -}}
{{- if eq $prefill $decode -}}
{{- fail "servingEngineSpec.modelSpec.pd.prefill and pd.decode must reference different roles" -}}
{{- end -}}
{{- $foundPrefill := false -}}{{- $foundDecode := false -}}
{{- $prefillReplicas := 0 -}}{{- $decodeReplicas := 0 -}}
{{- $maxPortSpan := 1 -}}
{{- range $role := $modelSpec.roles -}}
{{- if or (eq $role.name $prefill) (eq $role.name $decode) -}}
{{- $roleReplicasRaw := 1 -}}{{- if hasKey $role "replicas" -}}{{- $roleReplicasRaw = $role.replicas -}}{{- end -}}
{{- if or (kindIs "string" $roleReplicasRaw) (kindIs "bool" $roleReplicasRaw) (not (regexMatch "^[1-9][0-9]*$" (toString $roleReplicasRaw))) -}}
{{- fail (printf "roles[%s].replicas must be a positive integer" $role.name) -}}
{{- end -}}
{{- $args := default (default "" $modelSpec.vllmArgs) $role.vllmArgs -}}
{{- $span := int (include "chart.kvTransferPortSpan" (list $args)) -}}
{{- if gt $span $maxPortSpan -}}{{- $maxPortSpan = $span -}}{{- end -}}
{{- if and (get $capability "usesPort") (eq $role.name $decode) -}}
{{- $decodePp := int (include "chart.vllmArgInt" (list $args (list "--pipeline-parallel-size" "-pp") 1)) -}}
{{- if ne $decodePp 1 -}}
{{- fail (printf "roles[%s] decode pipeline parallel size must be 1 for %s, got %d" $role.name $connector $decodePp) -}}
{{- end -}}
{{- end -}}
{{- if eq $role.name $prefill -}}{{- $foundPrefill = true -}}{{- $prefillReplicas = int $roleReplicasRaw -}}{{- end -}}
{{- if eq $role.name $decode -}}{{- $foundDecode = true -}}{{- $decodeReplicas = int $roleReplicasRaw -}}{{- end -}}
{{- end -}}
{{- end -}}
{{- if not $foundPrefill -}}{{- fail (printf "pd.prefill=%q does not reference an existing roles[].name" $prefill) -}}{{- end -}}
{{- if not $foundDecode -}}{{- fail (printf "pd.decode=%q does not reference an existing roles[].name" $decode) -}}{{- end -}}
{{- if or (not (hasKey $kv "identity")) (not (kindIs "map" $kv.identity)) -}}
{{- fail "servingEngineSpec.modelSpec.pd.kvTransfer.identity is required and must be a map" -}}
{{- end -}}
{{- $identity := $kv.identity -}}
{{- range $field := keys $identity -}}
{{- if not (has $field (list "engineIdBase" "kvPortBase" "instanceStride")) -}}
{{- fail (printf "pd.kvTransfer.identity contains unsupported field %q" $field) -}}
{{- end -}}
{{- end -}}
{{- if not (hasKey $identity "engineIdBase") -}}
{{- fail "servingEngineSpec.modelSpec.pd.kvTransfer.identity.engineIdBase is required" -}}
{{- end -}}
{{- $engineBaseRaw := get $identity "engineIdBase" -}}
{{- if or (kindIs "string" $engineBaseRaw) (kindIs "bool" $engineBaseRaw) (not (regexMatch "^[0-9]+$" (toString $engineBaseRaw))) -}}
{{- fail "pd.kvTransfer.identity.engineIdBase must be an integer >= 0" -}}
{{- end -}}
{{- $groupsRaw := 1 -}}{{- if hasKey $modelSpec "replicas" -}}{{- $groupsRaw = $modelSpec.replicas -}}{{- end -}}
{{- if or (kindIs "string" $groupsRaw) (kindIs "bool" $groupsRaw) (not (regexMatch "^[1-9][0-9]*$" (toString $groupsRaw))) -}}
{{- fail "servingEngineSpec.modelSpec.replicas must be a positive integer" -}}
{{- end -}}
{{- if get $capability "usesPort" -}}
{{- range $field := list "kvPortBase" "instanceStride" -}}
{{- if not (hasKey $identity $field) -}}{{- fail (printf "pd.kvTransfer.identity.%s is required for %s" $field $connector) -}}{{- end -}}
{{- $raw := get $identity $field -}}
{{- if or (kindIs "string" $raw) (kindIs "bool" $raw) (not (regexMatch "^[1-9][0-9]*$" (toString $raw))) -}}
{{- fail (printf "pd.kvTransfer.identity.%s must be a positive integer" $field) -}}
{{- end -}}
{{- end -}}
{{- $portBase := int (get $identity "kvPortBase") -}}
{{- $stride := int (get $identity "instanceStride") -}}
{{- $minimumStride := 100 -}}{{- if gt $maxPortSpan $minimumStride -}}{{- $minimumStride = $maxPortSpan -}}{{- end -}}
{{- if or (lt $portBase 1) (gt $portBase 65535) -}}
{{- fail "pd.kvTransfer.identity.kvPortBase must be in 1..65535" -}}
{{- end -}}
{{- if lt $stride $minimumStride -}}
{{- fail (printf "pd.kvTransfer.identity.instanceStride=%d is too small; require at least %d for the configured DP x TP x PP x context-parallel span" $stride $minimumStride) -}}
{{- end -}}
{{- $configuredInstances := mul (int $groupsRaw) (add $prefillReplicas $decodeReplicas) -}}
{{- $lastPort := add $portBase (mul (sub $configuredInstances 1) $stride) (sub $maxPortSpan 1) -}}
{{- if gt $lastPort 65535 -}}
{{- fail (printf "pd.kvTransfer identity port range exceeds 65535: last configured port is %d" $lastPort) -}}
{{- end -}}
{{- else -}}
{{- if hasKey $identity "kvPortBase" -}}{{- fail (printf "pd.kvTransfer.identity.kvPortBase is not allowed for %s" $connector) -}}{{- end -}}
{{- if hasKey $identity "instanceStride" -}}{{- fail (printf "pd.kvTransfer.identity.instanceStride is not allowed for %s" $connector) -}}{{- end -}}
{{- end -}}
{{- $ucmEnabled := eq (include "chart.ucmConfigEnabled" (list $modelSpec)) "true" -}}
{{- if and $ucmEnabled (not (get $capability "supportsUcm")) -}}
{{- fail (printf "pd.kvTransfer.connector=%s cannot be combined with effective unifiedcacheConfig" $connector) -}}
{{- end -}}
{{- if and (eq (include "chart.pdMooncakeMasterRequested" (list $modelSpec)) "true") (ne $routerType "mooncake") -}}
{{- fail "pd.mooncake.master.enabled=true requires pd.kvTransfer.routerType=mooncake" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Returns "true" when automatic kvcs_store_id detection and runtime patching are enabled.
Defaults to true for backward compatibility.
*/}}
{{- define "chart.kvcsStoreIdAutoDetectEnabled" -}}
{{- $modelSpec := index . 0 -}}
{{- if and (hasKey $modelSpec "unifiedcacheConfig") (kindIs "map" $modelSpec.unifiedcacheConfig) (hasKey $modelSpec.unifiedcacheConfig "kvcsStoreIdAutoDetect") -}}
{{- ternary "true" "false" $modelSpec.unifiedcacheConfig.kvcsStoreIdAutoDetect -}}
{{- else if and (hasKey $modelSpec "unifiedcacheConfig") (kindIs "map" $modelSpec.unifiedcacheConfig) (hasKey $modelSpec.unifiedcacheConfig "autoDetectKvcsStoreId") -}}
{{- ternary "true" "false" $modelSpec.unifiedcacheConfig.autoDetectKvcsStoreId -}}
{{- else if and (hasKey $modelSpec "unifiedcacheConfig") (kindIs "map" $modelSpec.unifiedcacheConfig) (hasKey $modelSpec.unifiedcacheConfig "kvcsEnable") -}}
{{- ternary "true" "false" $modelSpec.unifiedcacheConfig.kvcsEnable -}}
{{- else -}}
true
{{- end -}}
{{- end -}}

{{/*
Join unified storage mount paths with ":".
*/}}
{{- define "chart.ucmStorageBackends" -}}
{{- $modelSpec := index . 0 -}}
{{- $storage := default (dict) $modelSpec.storage -}}
{{- $unifiedStorage := default (list) $storage.unifiedcacheStorage -}}
{{- $paths := list -}}
{{- range $s := $unifiedStorage -}}
{{- $paths = append $paths $s.mountPath -}}
{{- end -}}
{{- join ":" $paths -}}
{{- end -}}

{{/* ===================== kthena 集成 helper ===================== */}}

{{/*
chart.kthenaName: ModelServing/ModelServer/ModelRoute 共用名 = <release>-<modelSpec.name>
Usage: include "chart.kthenaName" (list $modelSpec $)
*/}}
{{- define "chart.kthenaName" -}}
{{- $modelSpec := index . 0 -}}
{{- $ := index . 1 -}}
{{- printf "%s-%s" $.Release.Name $modelSpec.name -}}
{{- end -}}

{{/*
chart.servedModelName: 模型对外名（API "model" 字段 / ModelRoute.modelName / ModelServer.model /
vllm --served-model-name 的统一来源）。取 modelSpec.modelName；为空则回退 modelSpec.modelPath。
Usage: include "chart.servedModelName" $modelSpec
*/}}
{{- define "chart.servedModelName" -}}
{{- default .modelPath .modelName -}}
{{- end -}}

{{/* ===================== PD KV-transfer-config 生成 ===================== */}}

{{/* chart.kvRoleOf: 由 pd.prefill/pd.decode 判某 role 是 producer/consumer/""（非 PD）。
Usage: include "chart.kvRoleOf" (list $role $modelSpec) */}}
{{- define "chart.kvRoleOf" -}}
{{- $role := index . 0 -}}
{{- $modelSpec := index . 1 -}}
{{- $pd := default (dict) $modelSpec.pd -}}
{{- if eq $role.name (default "" $pd.prefill) -}}producer
{{- else if eq $role.name (default "" $pd.decode) -}}consumer
{{- end -}}
{{- end -}}

{{/* ===================== 生命周期钩子（plan/vllm-lifecycle-hooks-2026-07-10.md） ===================== */}}

{{/* chart.validateHooksMap: 校验一个 hooks map（modelSpec.hooks 或 roles[].hooks）。
键仅允许 preStart/postReady/preStop（onExit 为 P2 预留，单独报「暂未支持」）；
值必须是脚本块字符串或 null（null=显式禁用，仅在 role 级覆盖 modelSpec 时有意义）。
Usage: include "chart.validateHooksMap" (list $hooks $where) */}}
{{- define "chart.validateHooksMap" -}}
{{- $hooks := index . 0 -}}
{{- $where := index . 1 -}}
{{- if $hooks -}}
{{- if not (kindIs "map" $hooks) -}}
{{- fail (printf "%s.hooks 必须是 map（键 preStart/postReady/preStop，值为脚本块字符串或 null）" $where) -}}
{{- end -}}
{{- range $k, $v := $hooks -}}
{{- if eq $k "onExit" -}}
{{- fail (printf "%s.hooks.onExit 暂未支持（P2 预留，见 plan/vllm-lifecycle-hooks-2026-07-10.md §2.2）" $where) -}}
{{- end -}}
{{- if not (has $k (list "preStart" "postReady" "preStop")) -}}
{{- fail (printf "%s.hooks 含未知钩子键 %q（仅支持 preStart/postReady/preStop）" $where $k) -}}
{{- end -}}
{{- if and (not (kindIs "invalid" $v)) (not (kindIs "string" $v)) -}}
{{- fail (printf "%s.hooks.%s 必须是脚本块字符串或 null（null=禁用该钩子）" $where $k) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* chart.resolveHooks: role.hooks 按键整体覆盖 modelSpec.hooks（hasKey 判定；显式 null=禁用），
输出 YAML map（钩子键→脚本），空白串视为未配置。值为纯字符串故无深拷贝问题；
禁止对 role.hooks/modelSpec.hooks 原对象 set/merge（.Values 会跨 entry/worker 多次渲染串值）。
本 helper 同时是 configmap-hooks.yaml 渲染与 modelserving projected source 追加的唯一判空来源
（两处条件漂移 = 引用不存在的 CM → Pod 永久 ContainerCreating）。
Usage: include "chart.resolveHooks" (list $role $modelSpec) | fromYaml */}}
{{- define "chart.resolveHooks" -}}
{{- $role := index . 0 -}}
{{- $modelSpec := index . 1 -}}
{{- include "chart.validateHooksMap" (list $modelSpec.hooks "servingEngineSpec.modelSpec") -}}
{{- include "chart.validateHooksMap" (list $role.hooks (printf "roles[%s]" (default "" $role.name))) -}}
{{- $roleHooks := default (dict) $role.hooks -}}
{{- $modelHooks := default (dict) $modelSpec.hooks -}}
{{- $out := dict -}}
{{- range $k := (list "preStart" "postReady" "preStop") -}}
{{- $v := "" -}}
{{- if hasKey $roleHooks $k -}}
{{- $v = index $roleHooks $k -}}
{{- else if hasKey $modelHooks $k -}}
{{- $v = index $modelHooks $k -}}
{{- end -}}
{{- if $v -}}
{{- if ne (trim $v) "" -}}
{{- $_ := set $out $k $v -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- toYaml $out -}}
{{- end -}}

{{/* chart.anyRoleHasHooks: 任一 role 解析后有钩子 → "true"。
args-entrypoint.sh 钩子块的模板 gate（无 hooks 时脚本必须逐字节不变）。
Usage: include "chart.anyRoleHasHooks" (list $modelSpec) */}}
{{- define "chart.anyRoleHasHooks" -}}
{{- $modelSpec := index . 0 -}}
{{- $any := false -}}
{{- range $role := $modelSpec.roles -}}
{{- if ne (trim (include "chart.resolveHooks" (list $role $modelSpec))) "{}" -}}
{{- $any = true -}}
{{- end -}}
{{- end -}}
{{- ternary "true" "false" $any -}}
{{- end -}}
