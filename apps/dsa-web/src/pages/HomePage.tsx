import type React from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getParsedApiError, type ParsedApiError } from '../api/error';
import { systemConfigApi } from '../api/systemConfig';
import { ApiErrorAlert, ConfirmDialog, Button, EmptyState, InlineAlert } from '../components/common';
import { DashboardStateBlock } from '../components/dashboard';
import { StockAutocomplete } from '../components/StockAutocomplete';
import { HistoryList } from '../components/history';
import { ReportMarkdown, ReportSummary } from '../components/report';
import { TaskPanel } from '../components/tasks';
import { useDashboardLifecycle, useHomeDashboardState } from '../hooks';
import type {
  SetupSmokeRunResponse,
  SetupWizardStatus,
  SystemConfigItem,
  TestLLMChannelResponse,
} from '../types/systemConfig';
import { getReportText, normalizeReportLanguage } from '../utils/reportLanguage';

const SETUP_PROMPT_STORAGE_KEY = 'dsa-first-run-setup-dismissed';
const MAX_SETUP_STOCKS = 3;

function splitCsv(value: string) {
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

function looksLikeStockCode(value: string) {
  const text = value.trim();
  return /^[A-Za-z]{1,5}$/.test(text) || /^[A-Za-z]{0,2}\d{3,6}(?:\.[A-Za-z]{2})?$/.test(text) || /^\d{5}\.HK$/i.test(text);
}

function buildSetupLLMPayload(items: SystemConfigItem[], maskToken: string) {
  const itemMap = new Map(items.map((item) => [item.key, String(item.value ?? '')]));
  const channelName = splitCsv(itemMap.get('LLM_CHANNELS') || '').find((name) => {
    const prefix = `LLM_${name.toUpperCase()}`;
    return (itemMap.get(`${prefix}_ENABLED`) || 'true').trim().toLowerCase() !== 'false';
  });
  if (channelName) {
    const prefix = `LLM_${channelName.toUpperCase()}`;
    return {
      name: channelName,
      protocol: itemMap.get(`${prefix}_PROTOCOL`) || 'openai',
      baseUrl: itemMap.get(`${prefix}_BASE_URL`) || '',
      apiKey: itemMap.get(`${prefix}_API_KEYS`) || itemMap.get(`${prefix}_API_KEY`) || '',
      models: splitCsv(itemMap.get(`${prefix}_MODELS`) || '').filter(Boolean),
      enabled: true,
      maskToken,
    };
  }
  if ((itemMap.get('GEMINI_API_KEY') || '').trim()) return { name: 'gemini', protocol: 'gemini', baseUrl: '', apiKey: itemMap.get('GEMINI_API_KEY') || '', models: ['gemini-2.5-flash'], enabled: true, maskToken };
  if ((itemMap.get('DEEPSEEK_API_KEY') || '').trim()) return { name: 'deepseek', protocol: 'deepseek', baseUrl: 'https://api.deepseek.com', apiKey: itemMap.get('DEEPSEEK_API_KEY') || '', models: ['deepseek-chat'], enabled: true, maskToken };
  if ((itemMap.get('OPENAI_API_KEY') || itemMap.get('AIHUBMIX_KEY') || '').trim()) return { name: 'openai', protocol: 'openai', baseUrl: itemMap.get('OPENAI_BASE_URL') || '', apiKey: itemMap.get('OPENAI_API_KEY') || itemMap.get('AIHUBMIX_KEY') || '', models: ['gpt-4o-mini'], enabled: true, maskToken };
  return null;
}

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [setupStatus, setSetupStatus] = useState<SetupWizardStatus | null>(null);
  const [setupError, setSetupError] = useState<ParsedApiError | null>(null);
  const [setupConfigVersion, setSetupConfigVersion] = useState('');
  const [setupItems, setSetupItems] = useState<SystemConfigItem[]>([]);
  const [setupMaskToken, setSetupMaskToken] = useState('******');
  const [setupStocks, setSetupStocks] = useState<string[]>([]);
  const [setupStockInput, setSetupStockInput] = useState('');
  const [setupStockError, setSetupStockError] = useState('');
  const [isSavingSetupStocks, setIsSavingSetupStocks] = useState(false);
  const [isTestingSetupLLM, setIsTestingSetupLLM] = useState(false);
  const [setupLLMResult, setSetupLLMResult] = useState<TestLLMChannelResponse | null>(null);
  const [isRunningSetupSmoke, setIsRunningSetupSmoke] = useState(false);
  const [setupSmokeResult, setSetupSmokeResult] = useState<SetupSmokeRunResponse | null>(null);
  const [setupDismissed, setSetupDismissed] = useState(() => (
    typeof localStorage !== 'undefined'
      ? localStorage.getItem(SETUP_PROMPT_STORAGE_KEY) === '1'
      : false
  ));

  const {
    query,
    inputError,
    duplicateError,
    error,
    isAnalyzing,
    historyItems,
    selectedHistoryIds,
    isDeletingHistory,
    isLoadingHistory,
    isLoadingMore,
    hasMore,
    selectedReport,
    isLoadingReport,
    activeTasks,
    markdownDrawerOpen,
    setQuery,
    clearError,
    loadInitialHistory,
    refreshHistory,
    loadMoreHistory,
    selectHistoryItem,
    toggleHistorySelection,
    toggleSelectAllVisible,
    deleteSelectedHistory,
    submitAnalysis,
    notify,
    setNotify,
    syncTaskCreated,
    syncTaskUpdated,
    syncTaskFailed,
    removeTask,
    openMarkdownDrawer,
    closeMarkdownDrawer,
    selectedIds,
  } = useHomeDashboardState();

  useEffect(() => {
    document.title = '每日选股分析 - DSA';
  }, []);

  const loadSetupStatus = useCallback(async () => {
    const payload = await systemConfigApi.getConfig(false);
    setSetupConfigVersion(payload.configVersion);
    setSetupStatus(payload.setupStatus);
    setSetupItems(payload.items);
    setSetupMaskToken(payload.maskToken);
    setSetupStocks(splitCsv(String(payload.items.find((item) => item.key === 'STOCK_LIST')?.value || '')).slice(0, MAX_SETUP_STOCKS));
    setSetupError(null);
    if (payload.setupStatus.isComplete && typeof localStorage !== 'undefined') {
      localStorage.removeItem(SETUP_PROMPT_STORAGE_KEY);
      setSetupDismissed(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    void loadSetupStatus().catch((error: unknown) => {
      if (active) setSetupError(getParsedApiError(error));
    });
    return () => {
      active = false;
    };
  }, [loadSetupStatus]);
  const reportLanguage = normalizeReportLanguage(selectedReport?.meta.reportLanguage);
  const reportText = getReportText(reportLanguage);
  const shouldShowSetupPrompt = Boolean(setupStatus && !setupStatus.isComplete && !setupDismissed);
  const setupLLMPayload = useMemo(() => buildSetupLLMPayload(setupItems, setupMaskToken), [setupItems, setupMaskToken]);

  const addSetupStock = useCallback((code: string, _name?: string, source?: 'manual' | 'autocomplete') => {
    const normalized = code.trim();
    if (!normalized) return;
    if (source !== 'autocomplete' && !looksLikeStockCode(normalized)) {
      setSetupStockError('名称输入请先从候选列表确认，避免写入错误股票。');
      return;
    }
    setSetupStocks((current) => {
      if (current.some((item) => item.toLowerCase() === normalized.toLowerCase())) return current;
      if (current.length >= MAX_SETUP_STOCKS) return current;
      return [...current, normalized];
    });
    setSetupStockInput('');
    setSetupStockError('');
    setSetupSmokeResult(null);
  }, []);

  const saveSetupStocks = useCallback(async () => {
    if (!setupStocks.length) {
      setSetupStockError('请先添加至少 1 只股票。');
      return;
    }
    setIsSavingSetupStocks(true);
    try {
      await systemConfigApi.update({
        configVersion: setupConfigVersion,
        maskToken: setupMaskToken,
        reloadNow: true,
        items: [{ key: 'STOCK_LIST', value: setupStocks.join(',') }],
      });
      await loadSetupStatus();
      setSetupStockError('');
    } catch (error: unknown) {
      setSetupStockError(getParsedApiError(error).message || '保存股票失败');
    } finally {
      setIsSavingSetupStocks(false);
    }
  }, [loadSetupStatus, setupConfigVersion, setupMaskToken, setupStocks]);

  const testSetupLLM = useCallback(async () => {
    if (!setupLLMPayload) {
      setSetupLLMResult({ success: false, message: '未检测到可测试的主模型配置', error: '请先在设置页配置 LLM', errorType: 'invalid_config', nextStep: '打开设置页补齐 AI 配置', stages: [] });
      return;
    }
    setIsTestingSetupLLM(true);
    try {
      setSetupLLMResult(await systemConfigApi.testLLMChannel(setupLLMPayload));
      await loadSetupStatus();
    } catch (error: unknown) {
      const parsed = getParsedApiError(error);
      setSetupLLMResult({ success: false, message: 'LLM 测试失败', error: parsed.message, errorType: 'network_error', nextStep: '请检查设置页中的渠道配置', stages: [] });
    } finally {
      setIsTestingSetupLLM(false);
    }
  }, [loadSetupStatus, setupLLMPayload]);

  const runSetupSmoke = useCallback(async () => {
    setIsRunningSetupSmoke(true);
    try {
      const result = await systemConfigApi.runSetupSmoke({ stockInput: setupStocks[0] || setupStockInput });
      setSetupSmokeResult(result);
      setSetupStatus(result.setupStatus);
    } catch (error: unknown) {
      const parsed = getParsedApiError(error);
      setSetupSmokeResult({ success: false, message: '首次试跑失败', errorCode: 'network_error', nextStep: '请稍后重试', summary: parsed.message, setupStatus: setupStatus || { isComplete: false, readyForSmoke: false, requiredMissingKeys: [], nextStepKey: null, checks: [] } });
    } finally {
      setIsRunningSetupSmoke(false);
    }
  }, [setupStatus, setupStockInput, setupStocks]);

  useDashboardLifecycle({
    loadInitialHistory,
    refreshHistory,
    syncTaskCreated,
    syncTaskUpdated,
    syncTaskFailed,
    removeTask,
  });

  const handleHistoryItemClick = useCallback((recordId: number) => {
    void selectHistoryItem(recordId);
    setSidebarOpen(false);
  }, [selectHistoryItem]);

  const handleSubmitAnalysis = useCallback(
    (
      stockCode?: string,
      stockName?: string,
      selectionSource?: 'manual' | 'autocomplete' | 'import' | 'image',
    ) => {
      void submitAnalysis({
        stockCode,
        stockName,
        originalQuery: query,
        selectionSource: selectionSource ?? 'manual',
      });
    },
    [query, submitAnalysis],
  );

  const handleAskFollowUp = useCallback(() => {
    if (selectedReport?.meta.id === undefined) {
      return;
    }

    const code = selectedReport.meta.stockCode;
    const name = selectedReport.meta.stockName;
    const rid = selectedReport.meta.id;
    navigate(`/chat?stock=${encodeURIComponent(code)}&name=${encodeURIComponent(name)}&recordId=${rid}`);
  }, [navigate, selectedReport]);

  const handleReanalyze = useCallback(() => {
    if (!selectedReport) {
      return;
    }

    void submitAnalysis({
      stockCode: selectedReport.meta.stockCode,
      stockName: selectedReport.meta.stockName,
      originalQuery: selectedReport.meta.stockCode,
      selectionSource: 'manual',
      forceRefresh: true,
    });
  }, [selectedReport, submitAnalysis]);

  const handleDeleteSelectedHistory = useCallback(() => {
    void deleteSelectedHistory();
    setShowDeleteConfirm(false);
  }, [deleteSelectedHistory]);

  const sidebarContent = useMemo(
    () => (
      <div className="flex min-h-0 h-full flex-col gap-3 overflow-hidden">
        <TaskPanel tasks={activeTasks} />
        <HistoryList
          items={historyItems}
          isLoading={isLoadingHistory}
          isLoadingMore={isLoadingMore}
          hasMore={hasMore}
          selectedId={selectedReport?.meta.id}
          selectedIds={selectedIds}
          isDeleting={isDeletingHistory}
          onItemClick={handleHistoryItemClick}
          onLoadMore={() => void loadMoreHistory()}
          onToggleItemSelection={toggleHistorySelection}
          onToggleSelectAll={toggleSelectAllVisible}
          onDeleteSelected={() => setShowDeleteConfirm(true)}
          className="flex-1 overflow-hidden"
        />
      </div>
    ),
    [
      activeTasks,
      hasMore,
      historyItems,
      isDeletingHistory,
      isLoadingHistory,
      isLoadingMore,
      handleHistoryItemClick,
      loadMoreHistory,
      selectedIds,
      selectedReport?.meta.id,
      toggleHistorySelection,
      toggleSelectAllVisible,
    ],
  );

  return (
    <div
      data-testid="home-dashboard"
      className="flex h-[calc(100vh-5rem)] w-full flex-col overflow-hidden md:flex-row sm:h-[calc(100vh-5.5rem)] lg:h-[calc(100vh-2rem)]"
    >
      <div className="flex-1 flex flex-col min-h-0 min-w-0 max-w-full lg:max-w-6xl mx-auto w-full">
        <header className="flex min-w-0 flex-shrink-0 items-center overflow-hidden px-3 py-3 md:px-4 md:py-4">
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2.5 md:flex-nowrap">
            <button
              onClick={() => setSidebarOpen(true)}
              className="md:hidden -ml-1 flex-shrink-0 rounded-lg p-1.5 text-secondary-text transition-colors hover:bg-hover hover:text-foreground"
              aria-label="历史记录"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="relative min-w-0 flex-1">
              <StockAutocomplete
                value={query}
                onChange={setQuery}
                onSubmit={(stockCode, stockName, selectionSource) => {
                  handleSubmitAnalysis(stockCode, stockName, selectionSource);
                }}
                placeholder="输入股票代码或名称，如 600519、贵州茅台、AAPL"
                disabled={isAnalyzing}
                className={inputError ? 'border-danger/50' : undefined}
              />
            </div>
            <label className="flex h-10 flex-shrink-0 cursor-pointer items-center gap-1.5 rounded-xl border border-subtle bg-surface/60 px-3 text-xs text-secondary-text select-none transition-colors hover:border-subtle-hover hover:text-foreground">
              <input
                type="checkbox"
                checked={notify}
                onChange={(e) => setNotify(e.target.checked)}
                className="h-3.5 w-3.5 rounded border-border accent-primary"
              />
              推送通知
            </label>
            <button
              type="button"
              onClick={() => handleSubmitAnalysis()}
              disabled={!query || isAnalyzing}
              className="btn-primary flex h-10 flex-shrink-0 items-center gap-1.5 whitespace-nowrap"
            >
              {isAnalyzing ? (
                <>
                  <svg className="h-3.5 w-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  分析中
                </>
              ) : (
                '分析'
              )}
            </button>
          </div>
        </header>

        {inputError || duplicateError ? (
          <div className="px-3 pb-2 md:px-4">
            {inputError ? (
              <InlineAlert
                variant="danger"
                title="输入有误"
                message={inputError}
                className="rounded-xl px-3 py-2 text-xs shadow-none"
              />
            ) : null}
            {!inputError && duplicateError ? (
              <InlineAlert
                variant="warning"
                title="任务已存在"
                message={duplicateError}
                className="rounded-xl px-3 py-2 text-xs shadow-none"
              />
            ) : null}
          </div>
        ) : null}

        {shouldShowSetupPrompt ? (
          <div className="px-3 pb-3 md:px-4">
            <div className="rounded-2xl border border-amber-400/30 bg-amber-50/80 px-4 py-4 dark:bg-amber-500/10">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <p className="text-sm font-semibold text-foreground">基础配置尚未完成</p>
                  <p className="mt-1 text-xs leading-6 text-secondary-text">
                    还缺 {setupStatus?.requiredMissingKeys.length || 0} 项关键配置：{
                      setupStatus?.checks
                        .filter((check) => setupStatus.requiredMissingKeys.includes(check.key))
                        .map((check) => check.title)
                        .join('、')
                    }。
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button type="button" variant="settings-primary" onClick={() => navigate('/settings')}>打开设置页</Button>
                  <Button type="button" variant="settings-secondary" onClick={() => { setSetupDismissed(true); if (typeof localStorage !== 'undefined') localStorage.setItem(SETUP_PROMPT_STORAGE_KEY, '1'); }}>稍后再说</Button>
                </div>
              </div>
              <div className="mt-3 grid gap-3 xl:grid-cols-[1.1fr_1fr]">
                <div className="rounded-xl border border-amber-400/20 bg-background/60 px-3 py-3">
                  <p className="text-xs font-medium text-foreground">当前检查项</p>
                  <p className="mt-2 text-xs leading-6 text-secondary-text">
                    {setupStatus?.checks.map((check) => `${check.title}：${check.message}`).join(' / ')}
                  </p>
                </div>
                <div className="space-y-3">
                  <div className="rounded-xl border border-amber-400/20 bg-background/60 px-3 py-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="text-xs text-secondary-text">
                        <p className="font-medium text-foreground">LLM 一键测试</p>
                        <p>{setupLLMPayload ? `${setupLLMPayload.name} · ${setupLLMPayload.models[0] || '未指定模型'}` : '请先到设置页补齐 AI 配置'}</p>
                      </div>
                      <Button type="button" variant="settings-secondary" disabled={isTestingSetupLLM} isLoading={isTestingSetupLLM} loadingText="测试中..." onClick={() => void testSetupLLM()}>测试 LLM</Button>
                    </div>
                    {setupLLMResult ? (
                      <div className="mt-2 text-xs leading-5">
                        <p className={setupLLMResult.success ? 'text-emerald-600 dark:text-emerald-300' : 'text-rose-600 dark:text-rose-300'}>{setupLLMResult.success ? 'LLM 可用' : `${setupLLMResult.errorType || 'unknown'}：${setupLLMResult.error || setupLLMResult.message}`}</p>
                        {setupLLMResult.stages[0] ? <p className="text-secondary-text">{setupLLMResult.stages.map((stage) => `${stage.title}：${stage.detail}`).join(' / ')}</p> : null}
                        {setupLLMResult.nextStep ? <p className="text-muted-text">下一步：{setupLLMResult.nextStep}</p> : null}
                      </div>
                    ) : null}
                  </div>
                  <div className="rounded-xl border border-amber-400/20 bg-background/60 px-3 py-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-xs font-medium text-foreground">保存 1-3 只试跑股票并执行 dry-run</p>
                      <div className="flex flex-wrap gap-2">
                        <Button type="button" variant="settings-secondary" disabled={isSavingSetupStocks} isLoading={isSavingSetupStocks} loadingText="保存中..." onClick={() => void saveSetupStocks()}>保存股票</Button>
                        <Button type="button" variant="settings-secondary" disabled={isRunningSetupSmoke} isLoading={isRunningSetupSmoke} loadingText="试跑中..." onClick={() => void runSetupSmoke()}>首次试跑</Button>
                      </div>
                    </div>
                    <div className="mt-2"><StockAutocomplete value={setupStockInput} onChange={setSetupStockInput} onSubmit={(code, name, source) => addSetupStock(code, name, source)} placeholder="输入 600519、腾讯、AAPL" disabled={setupStocks.length >= MAX_SETUP_STOCKS} /></div>
                    {!!setupStocks.length && (
                      <div className="mt-2 flex flex-wrap gap-2 text-xs">
                        {setupStocks.map((stock) => (
                          <button key={stock} type="button" className="rounded-full border border-subtle bg-background/70 px-3 py-1 text-secondary-text" onClick={() => setSetupStocks((current) => current.filter((item) => item !== stock))}>
                            {stock} ×
                          </button>
                        ))}
                      </div>
                    )}
                    {setupStockError ? <p className="mt-2 text-xs text-rose-600 dark:text-rose-300">{setupStockError}</p> : null}
                    {setupSmokeResult ? (
                      <div className="mt-2 text-xs leading-5">
                        <p className={setupSmokeResult.success ? 'text-emerald-600 dark:text-emerald-300' : 'text-rose-600 dark:text-rose-300'}>{setupSmokeResult.message}</p>
                        {setupSmokeResult.summary ? <p className="text-secondary-text">{setupSmokeResult.summary}</p> : null}
                        {setupSmokeResult.nextStep ? <p className="text-muted-text">下一步：{setupSmokeResult.nextStep}</p> : null}
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>
              {setupError ? <ApiErrorAlert className="mt-3" error={setupError} /> : null}
            </div>
          </div>
        ) : null}

        <div className="flex-1 flex min-h-0 overflow-hidden">
          <div className="hidden min-h-0 w-64 shrink-0 flex-col overflow-hidden pl-4 pb-4 md:flex lg:w-72">
            {sidebarContent}
          </div>

          {sidebarOpen ? (
            <div className="fixed inset-0 z-40 md:hidden" onClick={() => setSidebarOpen(false)}>
              <div className="page-drawer-overlay absolute inset-0" />
              <div
                className="dashboard-card absolute bottom-0 left-0 top-0 flex w-72 flex-col overflow-hidden !rounded-none !rounded-r-xl p-3 shadow-2xl"
                onClick={(event) => event.stopPropagation()}
              >
                {sidebarContent}
              </div>
            </div>
          ) : null}

          <section className="flex-1 min-w-0 min-h-0 overflow-x-auto overflow-y-auto px-3 pb-4 md:px-6 touch-pan-y">
            {error ? (
              <ApiErrorAlert
                error={error}
                className="mb-3"
                onDismiss={clearError}
              />
            ) : null}
            {isLoadingReport ? (
              <div className="flex h-full flex-col items-center justify-center">
                <DashboardStateBlock title="加载报告中..." loading />
              </div>
            ) : selectedReport ? (
              <div className="max-w-4xl space-y-4 pb-8">
                <div className="flex flex-wrap items-center justify-end gap-2">
                  <Button
                    variant="home-action-ai"
                    size="sm"
                    disabled={isAnalyzing || selectedReport.meta.id === undefined}
                    onClick={handleReanalyze}
                  >
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    {reportText.reanalyze}
                  </Button>
                  <Button
                    variant="home-action-ai"
                    size="sm"
                    disabled={selectedReport.meta.id === undefined}
                    onClick={handleAskFollowUp}
                  >
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    追问 AI
                  </Button>
                  <Button
                    variant="home-action-ai"
                    size="sm"
                    disabled={selectedReport.meta.id === undefined}
                    onClick={openMarkdownDrawer}
                  >
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    {reportText.fullReport}
                  </Button>
                </div>
                <ReportSummary data={selectedReport} isHistory />
              </div>
            ) : (
              <div className="flex h-full items-center justify-center">
                <EmptyState
                  title="开始分析"
                  description="输入股票代码进行分析，或从左侧选择历史报告查看。"
                  className="max-w-xl border-dashed"
                  icon={(
                    <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  )}
                />
              </div>
            )}
          </section>
        </div>
      </div>

      {markdownDrawerOpen && selectedReport?.meta.id ? (
        <ReportMarkdown
          recordId={selectedReport.meta.id}
          stockName={selectedReport.meta.stockName || ''}
          stockCode={selectedReport.meta.stockCode}
          reportLanguage={reportLanguage}
          onClose={closeMarkdownDrawer}
        />
      ) : null}

      <ConfirmDialog
        isOpen={showDeleteConfirm}
        title="删除历史记录"
        message={
          selectedHistoryIds.length === 1
            ? '确认删除这条历史记录吗？删除后将不可恢复。'
            : `确认删除选中的 ${selectedHistoryIds.length} 条历史记录吗？删除后将不可恢复。`
        }
        confirmText={isDeletingHistory ? '删除中...' : '确认删除'}
        cancelText="取消"
        isDanger={true}
        onConfirm={handleDeleteSelectedHistory}
        onCancel={() => setShowDeleteConfirm(false)}
      />
    </div>
  );
};

export default HomePage;
