<template>
  <div class="app-shell">
    <aside class="side-nav">
      <div class="brand">模型工具</div>
      <router-link to="/probe">模型探针</router-link>
      <router-link to="/detection">模型检测</router-link>
    </aside>
    <main class="main-panel">
      <router-view />
    </main>
  </div>
</template>
