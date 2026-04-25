export type SystemConfigCategory =
  | 'base'
  | 'data_source'
  | 'ai_model'
  | 'notification'
  | 'system'
  | 'agent'
  | 'backtest'
  | 'uncategorized';

export type SystemConfigDataType =
  | 'string'
  | 'integer'
  | 'number'
  | 'boolean'
  | 'array'
  | 'json'
  | 'time';

export type SystemConfigUIControl =
  | 'text'
  | 'password'
  | 'number'
  | 'select'
  | 'textarea'
  | 'switch'
  | 'time';

export interface SystemConfigOption {
  label: string;
  value: string;
}

export interface SystemConfigFieldSchema {
  key: string;
  title?: string;
  description?: string;
  category: SystemConfigCategory;
  dataType: SystemConfigDataType;
  uiControl: SystemConfigUIControl;
  isSensitive: boolean;
  isRequired: boolean;
  isEditable: boolean;
  defaultValue?: string | null;
  options: Array<string | SystemConfigOption>;
  validation: Record<string, unknown>;
  displayOrder: number;
}

export interface SystemConfigCategorySchema {
  category: SystemConfigCategory;
  title: string;
  description?: string;
  displayOrder: number;
  fields: SystemConfigFieldSchema[];
}

export interface SystemConfigSchemaResponse {
  schemaVersion: string;
  categories: SystemConfigCategorySchema[];
}

export interface SystemConfigItem {
  key: string;
  value: string;
  rawValueExists: boolean;
  isMasked: boolean;
  schema?: SystemConfigFieldSchema;
}

export interface SystemConfigResponse {
  configVersion: string;
  maskToken: string;
  items: SystemConfigItem[];
  updatedAt?: string;
  setupStatus: SetupWizardStatus;
}

export interface ExportSystemConfigResponse {
  content: string;
  configVersion: string;
  updatedAt?: string;
}

export interface SystemConfigUpdateItem {
  key: string;
  value: string;
}

export interface UpdateSystemConfigRequest {
  configVersion: string;
  maskToken?: string;
  reloadNow?: boolean;
  items: SystemConfigUpdateItem[];
}

export interface UpdateSystemConfigResponse {
  success: boolean;
  configVersion: string;
  appliedCount: number;
  skippedMaskedCount: number;
  reloadTriggered: boolean;
  updatedKeys: string[];
  warnings: string[];
}

export interface ValidateSystemConfigRequest {
  items: SystemConfigUpdateItem[];
}

export interface ImportSystemConfigRequest {
  configVersion: string;
  content: string;
  reloadNow?: boolean;
}

export interface ConfigValidationIssue {
  key: string;
  code: string;
  message: string;
  severity: 'error' | 'warning';
  expected?: string;
  actual?: string;
}

export interface ValidateSystemConfigResponse {
  valid: boolean;
  issues: ConfigValidationIssue[];
}

export interface TestLLMChannelRequest {
  name: string;
  protocol: string;
  baseUrl?: string;
  apiKey?: string;
  models: string[];
  enabled?: boolean;
  timeoutSeconds?: number;
  maskToken?: string;
}

export interface SetupWizardCheck {
  key: string;
  title: string;
  category: string;
  required: boolean;
  status: 'configured' | 'needs_action' | 'optional' | 'inherited' | 'warning';
  message: string;
  nextAction?: string | null;
}

export interface SetupWizardStatus {
  isComplete: boolean;
  readyForSmoke: boolean;
  requiredMissingKeys: string[];
  nextStepKey?: string | null;
  checks: SetupWizardCheck[];
}

export interface LLMTestStage {
  key: string;
  title: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  detail: string;
}

export interface TestLLMChannelResponse {
  success: boolean;
  message: string;
  error?: string | null;
  errorType?: string | null;
  resolvedModel?: string | null;
  latencyMs?: number | null;
  nextStep?: string | null;
  stages: LLMTestStage[];
}

export interface DiscoverLLMChannelModelsRequest {
  name: string;
  protocol: string;
  baseUrl?: string;
  apiKey?: string;
  models?: string[];
  timeoutSeconds?: number;
}

export interface DiscoverLLMChannelModelsResponse {
  success: boolean;
  message: string;
  error?: string | null;
  resolvedProtocol?: string | null;
  models: string[];
  latencyMs?: number | null;
}

export interface SetupSmokeRunRequest {
  stockInput?: string;
}

export interface SetupSmokeRunResponse {
  success: boolean;
  message: string;
  errorCode?: string | null;
  nextStep?: string | null;
  resolvedStockCode?: string | null;
  summary?: string | null;
  setupStatus: SetupWizardStatus;
}

export interface SystemConfigValidationErrorResponse {
  error: string;
  message: string;
  issues: ConfigValidationIssue[];
}

export interface SystemConfigConflictResponse {
  error: string;
  message: string;
  currentConfigVersion: string;
}
