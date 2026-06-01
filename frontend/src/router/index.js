import { createRouter, createWebHistory } from 'vue-router'
import Probe from '../views/Probe.vue'
import Detection from '../views/Detection.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/probe' },
    { path: '/probe', component: Probe },
    { path: '/detection', component: Detection },
  ],
})
