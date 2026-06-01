<template>
  <div class="detection-layout">
    <div class="detection-form-area">
      <h3 style="margin:0 0 20px 0">模型检测</h3>

      <el-form :model="runForm" label-width="90px" size="small" style="max-width:560px">
        <el-form-item label="Provider">
          <el-select
            v-model="runForm.config_id"
            placeholder="选择 Provider 配置"
            style="width:100%"
            @change="onProviderChange"
          >
            <el-option
              v-for="c in probeConfigs"
              :key="c.id"
              :label="c.label || c.base_url"
              :value="c.id"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="模型">
          <el-select
            v-model="runForm.models"
            multiple
            collapse-tags
            collapse-tags-tooltip
            placeholder="请先选择 Provider"
            style="width:100%"
            :disabled="!availableModels.length"
            :loading="modelsLoading"
          >
            <template #header>
              <div style="padding:4px 12px;display:flex;gap:8px">
                <el-button size="small" text @click="selectAllModels">全选</el-button>
                <el-button size="small" text @click="runForm.models = []">清空</el-button>
              </div>
            </template>
            <el-option
              v-for="m in availableModels"
              :key="m"
              :label="m"
              :value="m"
            />
          </el-select>
          <div v-if="runForm.config_id && !modelsLoading && !availableModels.length" class="hint-text">
            该 Provider 暂无探测记录，请先在探针页面探测一次
          </div>
        </el-form-item>

        <el-form-item label="检测套件">
          <el-select v-model="runForm.suite" style="width:100%">
            <el-option
              v-for="s in suites"
              :key="s.value"
              :label="s.label"
              :value="s.value"
            />
          </el-select>
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            :loading="running"
            :disabled="!runForm.config_id || !runForm.models.length || !runForm.suite"
            @click="startRun"
          >
            {{ running ? '检测中...' : '开始检测' }}
          </el-button>
          <span v-if="runForm.models.length" style="margin-left:12px;font-size:12px;color:#909399">
            已选 {{ runForm.models.length }} 个模型
          </span>
        </el-form-item>
      </el-form>
    </div>

    <el-divider />

    <!-- 检测历史 -->
    <div class="history-section">
      <div style="font-size:14px;font-weight:500;margin-bottom:12px">检测历史</div>
      <el-table :data="runs" size="small" @row-click="showDetail" style="cursor:pointer" stripe>
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column label="Provider" min-width="140" show-overflow-tooltip>
          <template #default="{ row }">{{ configLabel(row.config_id) }}</template>
        </el-table-column>
        <el-table-column prop="suite" label="套件" width="120" />
        <el-table-column label="状态" width="80">
          <template #default="{ row }">
            <el-tag
              :type="runStatusTag(row.status)"
              size="small"
            >{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="得分" width="80" align="right">
          <template #default="{ row }">
            <span v-if="row.summary?.score != null" :style="scoreColor(row.summary.score)">
              {{ row.summary.score }}
            </span>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="通过/总数" width="100" align="right">
          <template #default="{ row }">
            <span v-if="row.summary">{{ row.summary.passed }}/{{ row.summary.total }}</span>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="started_at" label="开始时间" min-width="160" />
        <el-table-column prop="finished_at" label="结束时间" min-width="160" />
        <el-table-column label="报告" width="90" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              size="small"
              text
              type="primary"
              :disabled="row.status !== 'done'"
              @click.stop="downloadReport(row)"
            >
              下载
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 详情弹窗 -->
    <el-dialog v-model="detailVisible" title="检测详情" width="80%" destroy-on-close>
      <div v-if="detail">
        <div style="margin-bottom:12px;display:flex;gap:24px;font-size:13px;color:#606266">
          <span>套件：{{ detail.suite }}</span>
          <span>状态：{{ statusLabel(detail.status) }}</span>
          <span v-if="detail.summary">
            通过 {{ detail.summary.passed }}/{{ detail.summary.total }}，得分 {{ detail.summary.score }}
          </span>
        </div>
        <el-table :data="detail.results || []" size="small" stripe max-height="500">
          <el-table-column prop="test_name" label="测试项" min-width="180" show-overflow-tooltip />
          <el-table-column label="结果" width="70" align="center">
            <template #default="{ row: r }">
              <el-tag :type="resultStatusTag(r.status)" size="small">
                {{ r.status }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="eval_type" label="类型" width="80" align="center" />
          <el-table-column prop="score" label="得分" width="70" align="right" />
          <el-table-column prop="latency_ms" label="延迟(ms)" width="100" align="right" />
          <el-table-column prop="message" label="信息" min-width="200" show-overflow-tooltip />
        </el-table>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../utils/api'

const suites = ref([])
const probeConfigs = ref([])
const runs = ref([])
const availableModels = ref([])
const modelsLoading = ref(false)
const runForm = ref({ config_id: '', suite: 'availability', models: [] })
const running = ref(false)
const detailVisible = ref(false)
const detail = ref(null)

onMounted(async () => {
  try {
    const [sResp, cResp, rResp] = await Promise.all([
      api.get('/api/detection/suites'),
      api.get('/api/probe/configs'),
      api.get('/api/detection/runs'),
    ])
    suites.value = sResp.data || []
    probeConfigs.value = Array.isArray(cResp.data) ? cResp.data : []
    runs.value = Array.isArray(rResp.data) ? rResp.data : []
  } catch (e) { /* */ }
})

async function onProviderChange(configId) {
  runForm.value.models = []
  availableModels.value = []
  if (!configId) return
  modelsLoading.value = true
  try {
    // 取该 Provider 最近一次探测记录的模型列表
    const { data: runs } = await api.get(`/api/probe/runs?config_id=${configId}&limit=1`)
    if (runs && runs.length > 0) {
      const { data: results } = await api.get(`/api/probe/runs/${runs[0].id}/results`)
      availableModels.value = (results || []).map(r => r.model).filter(Boolean).sort()
    }
  } catch (e) { /* */ }
  finally { modelsLoading.value = false }
}

function selectAllModels() {
  runForm.value.models = [...availableModels.value]
}

async function startRun() {
  if (!runForm.value.config_id) { ElMessage.warning('请选择 Provider'); return }
  if (!runForm.value.models.length) { ElMessage.warning('请至少选择一个模型'); return }
  running.value = true
  try {
    await api.post('/api/detection/run', {
      config_id: runForm.value.config_id,
      suite: runForm.value.suite,
      models: runForm.value.models,
    })
    ElMessage.success('检测完成')
    const { data } = await api.get('/api/detection/runs')
    runs.value = Array.isArray(data) ? data : []
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '检测失败')
  } finally {
    running.value = false
  }
}

async function showDetail(row) {
  try {
    const { data } = await api.get(`/api/detection/runs/${row.id}`)
    detail.value = data
    detailVisible.value = true
  } catch (e) { /* */ }
}

async function downloadReport(row) {
  try {
    const resp = await api.get(`/api/detection/runs/${row.id}/report`, { responseType: 'blob' })
    const blob = new Blob([resp.data], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    const disposition = resp.headers?.['content-disposition'] || ''
    link.href = url
    link.download = parseFilename(disposition) || `TokenStar_${row.suite || '模型检测'}_报告_${row.id}.md`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '报告下载失败')
  }
}

function parseFilename(disposition) {
  const utf8 = disposition.match(/filename\*=UTF-8''([^;]+)/)
  if (utf8) return decodeURIComponent(utf8[1])
  const normal = disposition.match(/filename="([^"]+)"/)
  return normal?.[1] || ''
}

function configLabel(configId) {
  const c = probeConfigs.value.find(x => x.id === configId)
  return c ? (c.label || c.base_url) : configId
}

function statusLabel(s) {
  return { done: '完成', running: '运行中', error: '错误' }[s] || s
}

function runStatusTag(s) {
  return { done: 'success', running: 'warning', error: 'danger' }[s] || 'info'
}

function resultStatusTag(s) {
  return { PASS: 'success', FAIL: 'danger', WARN: 'warning', INFO: 'info' }[s] || 'info'
}

function scoreColor(score) {
  if (score >= 80) return 'color:#67c23a;font-weight:600'
  if (score >= 50) return 'color:#e6a23c;font-weight:600'
  return 'color:#f56c6c;font-weight:600'
}
</script>

<style scoped>
.detection-layout { padding: 20px; overflow-y: auto; height: 100%; }
.detection-form-area { max-width: 560px; }
.history-section { width: 100%; max-width: 100%; }
.hint-text { font-size: 12px; color: #e6a23c; margin-top: 4px; }
</style>
