{{/*
Expand the name of the chart.
*/}}
{{- define "statewave.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "statewave.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label.
*/}}
{{- define "statewave.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "statewave.labels" -}}
helm.sh/chart: {{ include "statewave.chart" . }}
{{ include "statewave.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: statewave
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "statewave.selectorLabels" -}}
app.kubernetes.io/name: {{ include "statewave.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
ServiceAccount name.
*/}}
{{- define "statewave.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "statewave.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Image reference. Falls back to .Chart.AppVersion when image.tag is empty.
*/}}
{{- define "statewave.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag }}
{{- printf "%s:%s" .Values.image.repository $tag }}
{{- end }}

{{/*
Name of the chart-managed Secret holding inline credentials. We only
materialise this Secret when at least one inline credential value is
non-empty, so operators using existingSecret references never see an
empty secret in their cluster.
*/}}
{{- define "statewave.inlineSecretName" -}}
{{- printf "%s-credentials" (include "statewave.fullname" .) }}
{{- end }}

{{/*
Whether the chart should render an inline credentials Secret.
True when any of database.url / llm.apiKey / auth.apiKey is supplied
inline (non-empty) AND not delegated to an existingSecret.
*/}}
{{- define "statewave.shouldRenderInlineSecret" -}}
{{- $needs := false -}}
{{- if and .Values.database.url (not .Values.database.existingSecret) -}}{{- $needs = true -}}{{- end -}}
{{- if and .Values.llm.apiKey (not .Values.llm.existingSecret) -}}{{- $needs = true -}}{{- end -}}
{{- if and .Values.auth.apiKey (not .Values.auth.existingSecret) -}}{{- $needs = true -}}{{- end -}}
{{- $needs -}}
{{- end -}}

{{/*
Common environment block — emitted by both the Deployment and the
migration Job so that DB credentials and provider config stay aligned.
*/}}
{{- define "statewave.commonEnv" -}}
# Database
- name: STATEWAVE_DATABASE_URL
  valueFrom:
    secretKeyRef:
      {{- if .Values.database.existingSecret }}
      name: {{ .Values.database.existingSecret }}
      key: {{ .Values.database.existingSecretKey }}
      {{- else }}
      name: {{ include "statewave.inlineSecretName" . }}
      key: STATEWAVE_DATABASE_URL
      {{- end }}
{{- end }}

{{/*
Runtime environment — only emitted by the Deployment. Pulls in the
common block plus everything the API process needs at request time.
The migration Job intentionally only needs the DB URL.
*/}}
{{- define "statewave.runtimeEnv" -}}
{{ include "statewave.commonEnv" . }}
- name: STATEWAVE_HOST
  value: "0.0.0.0"
- name: STATEWAVE_PORT
  value: "8100"

# Compiler + embedding
- name: STATEWAVE_COMPILER_TYPE
  value: {{ .Values.compiler.type | quote }}
- name: STATEWAVE_EMBEDDING_PROVIDER
  value: {{ .Values.embedding.provider | quote }}
- name: STATEWAVE_EMBEDDING_DIMENSIONS
  value: {{ .Values.embedding.dimensions | quote }}

# LiteLLM (only meaningful when compiler=llm or embedding=litellm)
- name: STATEWAVE_LITELLM_MODEL
  value: {{ .Values.llm.model | quote }}
- name: STATEWAVE_LITELLM_EMBEDDING_MODEL
  value: {{ .Values.llm.embeddingModel | quote }}
{{- if .Values.llm.apiBase }}
- name: STATEWAVE_LITELLM_API_BASE
  value: {{ .Values.llm.apiBase | quote }}
{{- end }}
- name: STATEWAVE_LITELLM_TIMEOUT_SECONDS
  value: {{ .Values.llm.timeoutSeconds | quote }}
- name: STATEWAVE_LITELLM_MAX_RETRIES
  value: {{ .Values.llm.maxRetries | quote }}
- name: STATEWAVE_LITELLM_TEMPERATURE
  value: {{ .Values.llm.temperature | quote }}
{{- if or .Values.llm.apiKey .Values.llm.existingSecret }}
- name: STATEWAVE_LITELLM_API_KEY
  valueFrom:
    secretKeyRef:
      {{- if .Values.llm.existingSecret }}
      name: {{ .Values.llm.existingSecret }}
      key: {{ .Values.llm.existingSecretKey }}
      {{- else }}
      name: {{ include "statewave.inlineSecretName" . }}
      key: STATEWAVE_LITELLM_API_KEY
      {{- end }}
{{- end }}

# API auth
{{- if or .Values.auth.apiKey .Values.auth.existingSecret }}
- name: STATEWAVE_API_KEY
  valueFrom:
    secretKeyRef:
      {{- if .Values.auth.existingSecret }}
      name: {{ .Values.auth.existingSecret }}
      key: {{ .Values.auth.existingSecretKey }}
      {{- else }}
      name: {{ include "statewave.inlineSecretName" . }}
      key: STATEWAVE_API_KEY
      {{- end }}
{{- end }}

# Rate limit + CORS
- name: STATEWAVE_RATE_LIMIT_RPM
  value: {{ .Values.rateLimit.rpm | quote }}
- name: STATEWAVE_CORS_ORIGINS
  value: {{ .Values.cors.origins | quote }}

# Webhooks
{{- if .Values.webhooks.url }}
- name: STATEWAVE_WEBHOOK_URL
  value: {{ .Values.webhooks.url | quote }}
- name: STATEWAVE_WEBHOOK_TIMEOUT
  value: {{ .Values.webhooks.timeoutSeconds | quote }}
{{- end }}

# Multi-tenant
{{- if .Values.multiTenant.enabled }}
- name: STATEWAVE_TENANT_HEADER
  value: {{ .Values.multiTenant.header | quote }}
- name: STATEWAVE_REQUIRE_TENANT
  value: {{ .Values.multiTenant.required | quote }}
{{- end }}

# Support pack — off by default for self-hosted operators
- name: STATEWAVE_AUTO_UPDATE_SUPPORT_PACK
  value: {{ .Values.supportPack.autoUpdate | quote }}
- name: STATEWAVE_BOOTSTRAP_DOCS_PACK
  value: {{ .Values.supportPack.bootstrapFromMountedDocs | quote }}

{{- with .Values.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end }}
