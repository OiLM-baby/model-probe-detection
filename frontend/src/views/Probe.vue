<template>
  <div class="probe-layout">
    <!-- 左侧：配置管理 -->
    <aside class="probe-left">
      <div class="panel-title">探针配置</div>

      <el-form :model="form" label-position="top" size="small">
        <el-form-item label="标签">
          <el-input v-model="form.label" placeholder="如 openai-proxy" />
        </el-form-item>
        <el-form-item label="Base URL">
          <el-input v-model="form.base_url" placeholder="https://api.openai.com" />
        </el-form-item>
        <el-form-item label="API Key">
          <el-input v-model="form.api_key" type="password" show-password />
        </el-form-item>
        <el-form-item label="API 格式（可选）">
          <el-select v-model="form.api_format" style="width:100%" clearable placeholder="不填默认 OpenAI">
            <el-option label="OpenAI" value="openai" />
            <el-option label="Anthropic" value="anthropic" />
            <el-option label="Responses" value="responses" />
          </el-select>
        </el-form-item>
        <el-button type="primary" style="width:100%" size="small" @click="saveConfig">保存配置</el-button>
      </el-form>

      <el-divider style="margin:16px 0" />

      <div class="config-list">
        <div
          v-for="cfg in configs"
          :key="cfg.id"
          class="config-item"
          :class="{ active: activeConfigId === cfg.id }"
          @click="selectConfig(cfg)"
        >
          <div class="config-item-top">
            <span class="config-label">{{ cfg.label || '未命名' }}</span>
            <el-tag size="small" type="info">{{ cfg.api_format || 'openai' }}</el-tag>
          </div>
          <div class="config-item-url">{{ cfg.base_url }}</div>
          <el-button size="small" type="danger" text style="margin-top:4px" @click.stop="deleteConfig(cfg.id)">删除</el-button>
        </div>
        <el-empty v-if="!configs.length" description="暂无配置" :image-size="50" />
      </div>
    </aside>

    <!-- 右侧：探测区 -->
    <main class="probe-right">
      <div class="probe-toolbar">
        <div class="toolbar-left">
          <span class="toolbar-title">{{ activeConfig ? (activeConfig.label || activeConfig.base_url) : '选择配置后开始探测' }}</span>
          <span v-if="probeTime && !probing" class="probe-time">探测于 {{ probeTime }}</span>
        </div>
        <div class="toolbar-right">
          <el-select v-model="probeType" size="small" style="width:190px" :disabled="probing">
            <el-option-group
              v-for="group in probeTypeGroups"
              :key="group.label"
              :label="group.label"
            >
              <el-option v-for="item in group.options" :key="item.value" :label="item.label" :value="item.value" />
            </el-option-group>
          </el-select>
          <el-button type="success" :loading="probing" :disabled="!activeConfig || selectedModels.length === 0" @click="runProbe(false)">
            {{ probing ? '探测中...' : `探测选中 (${selectedModels.length})` }}
          </el-button>
          <el-button type="primary" :loading="probing" :disabled="!activeConfig" @click="runProbe(true)">
            {{ probing ? '探测中...' : '全量探测' }}
          </el-button>
        </div>
      </div>

      <!-- 进度提示 -->
      <div v-if="probing" class="probing-hint">
        <el-icon class="spin"><Loading /></el-icon>
        <span v-if="totalModels > 0">已完成 {{ completedCount }} / {{ totalModels }} 个模型</span>
        <span v-else>正在获取模型列表...</span>
      </div>

      <!-- 汇总 -->
      <div v-if="displayResults.length > 0 || isDone" class="probe-summary">
        <el-statistic title="模型总数" :value="totalModels || displayResults.length" />
        <el-statistic title="可用">
          <template #number>
            <span style="color:#67c23a;font-size:24px;font-weight:600">{{ passedCount }}</span>
          </template>
        </el-statistic>
        <el-statistic title="不可用">
          <template #number>
            <span :style="failedCount > 0 ? 'color:#f56c6c' : ''" style="font-size:24px;font-weight:600">{{ failedCount }}</span>
          </template>
        </el-statistic>
        <el-statistic v-if="skippedCount > 0" title="跳过">
          <template #number>
            <span style="color:#909399;font-size:24px;font-weight:600">{{ skippedCount }}</span>
          </template>
        </el-statistic>
        <el-statistic v-if="displayResults.length > 0" title="平均延迟" :value="avgLatencyMs" suffix="ms" />
      </div>

      <!-- 错误提示 -->
      <el-alert
        v-if="listError"
        :title="`模型列表获取失败${listErrorCategory ? `（${listErrorCategory}）` : ''}: ${listError}`"
        type="warning"
        :closable="false"
        show-icon
      />

      <!-- 结果表格 -->
      <el-table
        v-if="displayResults.length > 0"
        :data="sortedResults"
        size="small"
        stripe
        style="width:100%"
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="55" :selectable="row => row.ok" />
        <el-table-column label="状态" width="70" align="center">
          <template #default="{ row }">
            <el-tag :type="row.ok ? 'success' : (row.skipped ? 'info' : 'danger')" size="small">
              {{ row.ok ? 'OK' : (row.skipped ? 'SKIP' : 'FAIL') }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column prop="model" label="模型" min-width="180" show-overflow-tooltip />
        <el-table-column prop="endpoint" label="接口" width="150" show-overflow-tooltip />

        <el-table-column label="失败原因" width="110" align="center">
          <template #default="{ row }">
            <el-tag v-if="!row.ok && row.error_category" :type="errorCategoryTag(row.error_category)" size="small">
              {{ row.error_category }}
            </el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <el-table-column :label="probeType === 'chat' ? '首 Token' : '首响应'" prop="first_token_ms" width="100" align="right">
          <template #default="{ row }">
            <span :class="latencyClass(row.first_token_ms)">
              {{ row.first_token_ms != null ? row.first_token_ms + 'ms' : '-' }}
            </span>
          </template>
        </el-table-column>

        <el-table-column label="总耗时" prop="latency_ms" width="90" align="right">
          <template #default="{ row }">{{ row.latency_ms }}ms</template>
        </el-table-column>

        <el-table-column v-if="probeType === 'chat'" label="吐字速度" prop="chars_per_second" width="100" align="right">
          <template #default="{ row }">
            {{ row.chars_per_second > 0 ? row.chars_per_second + ' c/s' : '-' }}
          </template>
        </el-table-column>

        <el-table-column label="响应预览 / 错误" min-width="200" show-overflow-tooltip>
          <template #default="{ row }">
            <span v-if="row.ok" class="preview-text">{{ row.response_preview }}</span>
            <span v-else class="error-text">{{ row.error }}</span>
          </template>
        </el-table-column>
      </el-table>

      <el-empty v-if="!displayResults.length && !probing" description="选择左侧配置，点击「重新探测」" :image-size="100" style="margin-top:60px" />
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
import api from '../utils/api'

const configs = ref([])
const activeConfig = ref(null)
const activeConfigId = ref('')
const probing = ref(false)
const results = ref([])
const totalModels = ref(0)
const listError = ref('')
const listErrorCategory = ref('')
const isDone = ref(false)
const probeTime = ref('')
const selectedModels = ref([])
const probeType = ref('chat')
let streamSeq = 0
let abortController = null

const form = ref({ label: '', base_url: '', api_key: '', api_format: '' })
const probeTypeGroups = [
  {
    label: '文本',
    options: [
      { label: '聊天', value: 'chat' },
    ],
  },
  {
    label: '多模态',
    options: [
      { label: '文生图', value: 'image_generation' },
      { label: '图生图', value: 'image_edit' },
      { label: 'Responses 图片', value: 'responses_image' },
      { label: 'Gemini/Banna 图片', value: 'banna_image' },
    ],
  },
]

const displayResults = computed(() => {
  const map = new Map()
  for (const item of results.value) {
    if (item?.model) map.set(item.model, item)
  }
  return [...map.values()]
})
const completedCount = computed(() => displayResults.value.length)
const passedCount = computed(() => displayResults.value.filter(r => r.ok).length)
const skippedCount = computed(() => displayResults.value.filter(r => r.skipped).length)
const failedCount = computed(() => displayResults.value.filter(r => !r.ok && !r.skipped).length)
const sortedResults = computed(() => [...displayResults.value].sort((a, b) => a.model.localeCompare(b.model)))
const avgLatencyMs = computed(() => {
  const done = displayResults.value.filter(r => r.latency_ms > 0)
  if (!done.length) return 0
  return Math.round(done.reduce((sum, r) => sum + r.latency_ms, 0) / done.length)
})

onMounted(() => loadConfigs())
onBeforeUnmount(() => stopActiveProbe())
watch(probeType, () => {
  selectedModels.value = []
  if (activeConfig.value) selectConfig(activeConfig.value)
})

async function loadConfigs() {
  try {
    const { data } = await api.get('/api/probe/configs')
    configs.value = Array.isArray(data) ? data : []
  } catch (e) { /* */ }
}

async function saveConfig() {
  if (!form.value.base_url) { ElMessage.warning('Base URL 不能为空'); return }
  try {
    await api.post('/api/probe/configs', form.value)
    form.value = { label: '', base_url: '', api_key: '', api_format: '' }
    await loadConfigs()
    ElMessage.success('配置已保存')
  } catch (e) {
    ElMessage.error('保存失败')
  }
}

async function deleteConfig(id) {
  try {
    await api.delete(`/api/probe/configs/${id}`)
    if (activeConfigId.value === id) { activeConfig.value = null; activeConfigId.value = ''; resetResults() }
    await loadConfigs()
    ElMessage.success('已删除')
  } catch (e) { ElMessage.error('删除失败') }
}

async function selectConfig(cfg) {
  stopActiveProbe()
  activeConfig.value = cfg
  activeConfigId.value = cfg.id
  resetResults()
  // 加载最近一次探测记录
  try {
    const { data: runs } = await api.get(`/api/probe/runs?config_id=${cfg.id}&limit=1&probe_type=${probeType.value}`)
    if (runs && runs.length > 0) {
      const run = runs[0]
      probeTime.value = run.created_at
      totalModels.value = run.total
      isDone.value = true
      const { data: rows } = await api.get(`/api/probe/runs/${run.id}/results`)
      results.value = rows || []
    }
  } catch (e) { /* 没有历史记录，静默忽略 */ }
}

function resetResults() {
  results.value = []
  totalModels.value = 0
  listError.value = ''
  listErrorCategory.value = ''
  isDone.value = false
  probeTime.value = ''
}

function stopActiveProbe() {
  streamSeq += 1
  if (abortController) {
    abortController.abort()
    abortController = null
  }
  probing.value = false
}

async function runProbe(isFullProbe = true) {
  if (!activeConfig.value || probing.value) return
  if (!isFullProbe && selectedModels.value.length === 0) {
    ElMessage.warning('请先选择要探测的模型')
    return
  }
  stopActiveProbe()
  const seq = ++streamSeq
  abortController = new AbortController()
  probing.value = true
  resetResults()

  const body = {
    config_id: activeConfig.value.id,
    base_url: activeConfig.value.base_url,
    api_key: activeConfig.value.api_key || '',
    api_format: activeConfig.value.api_format || 'openai',
    probe_type: probeType.value,
  }

  // 如果不是全量探测，添加选中的模型列表
  if (!isFullProbe) {
    body.models = selectedModels.value
  }

  try {
    const response = await fetch('/api/probe/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: abortController.signal,
    })

    if (seq !== streamSeq) return

    if (!response.ok) {
      const err = await response.json().catch(() => ({}))
      ElMessage.error(err.detail || '探测失败')
      probing.value = false
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (seq !== streamSeq) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const d = JSON.parse(line.slice(6))
          if (seq !== streamSeq) return
          if (d.type === 'models') {
            totalModels.value = d.total
            results.value = []
          } else if (d.type === 'result') {
            upsertResult(d)
          } else if (d.type === 'done') {
            isDone.value = true
            totalModels.value = d.total || totalModels.value
            // 探测完成后记录当前时间
            probeTime.value = new Date().toLocaleString('zh-CN', { hour12: false })
          } else if (d.type === 'error') {
            listError.value = d.message
            listErrorCategory.value = d.error_category || ''
          }
        } catch (e) { /* */ }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') ElMessage.error('连接失败: ' + e.message)
  } finally {
    if (seq === streamSeq) {
      probing.value = false
      abortController = null
    }
  }
}

function upsertResult(row) {
  if (!row?.model) return
  const index = results.value.findIndex(item => item.model === row.model)
  if (index >= 0) {
    results.value[index] = row
  } else {
    results.value.push(row)
  }
}

function handleSelectionChange(selection) {
  selectedModels.value = selection.map(row => row.model)
}

function latencyClass(ms) {
  if (ms == null) return ''
  if (ms < 800) return 'latency-good'
  if (ms < 2000) return 'latency-warn'
  return 'latency-bad'
}

function errorCategoryTag(category) {
  if (['鉴权失败', '拒绝访问', '额度不足'].includes(category)) return 'danger'
  if (['限流', '上游异常', '超时'].includes(category)) return 'warning'
  if (category === '模型不可用') return 'info'
  return 'danger'
}
</script>

<style scoped>
.probe-layout {
  display: grid;
  grid-template-columns: 260px 1fr;
  height: 100%;
  overflow: hidden;
}
.probe-left {
  padding: 16px;
  border-right: 1px solid #e4e7ed;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}
.panel-title { font-size: 15px; font-weight: 600; margin-bottom: 16px; }
.config-list { flex: 1; overflow-y: auto; }
.config-item {
  padding: 10px 12px;
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.2s;
}
.config-item:hover { border-color: #409eff; background: #f0f7ff; }
.config-item.active { border-color: #409eff; background: #ecf5ff; }
.config-item-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.config-label { font-size: 13px; font-weight: 500; }
.config-item-url { font-size: 11px; color: #909399; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.probe-right { padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
.probe-toolbar { display: flex; justify-content: space-between; align-items: center; }
.toolbar-left { display: flex; flex-direction: column; gap: 2px; }
.toolbar-right { display: flex; gap: 8px; }
.toolbar-title { font-size: 15px; font-weight: 500; color: #303133; }
.probe-time { font-size: 12px; color: #909399; }

.probing-hint { display: flex; align-items: center; gap: 8px; color: #409eff; font-size: 14px; }
.spin { animation: spin 1s linear infinite; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.probe-summary { display: flex; gap: 32px; padding: 16px 20px; background: #f5f7fa; border-radius: 8px; }

.latency-good { color: #67c23a; font-weight: 500; }
.latency-warn { color: #e6a23c; font-weight: 500; }
.latency-bad  { color: #f56c6c; font-weight: 500; }
.preview-text { color: #606266; font-size: 12px; }
.error-text   { color: #f56c6c; font-size: 12px; }
</style>
