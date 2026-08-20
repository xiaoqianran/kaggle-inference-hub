export const TRIPO_RESOLUTION_OPTIONS = [
  {
    value: 128,
    name: '快速',
    shortLabel: '128 · 快速',
    description: '粗略网格，适合快速试效果；速度最快、显存最低',
  },
  {
    value: 256,
    name: '标准',
    shortLabel: '256 · 标准',
    description: '细节和速度平衡；日常使用推荐',
  },
  {
    value: 384,
    name: '精细',
    shortLabel: '384 · 精细',
    description: '更平滑的模型轮廓；耗时和显存更高',
  },
  {
    value: 512,
    name: '高显存',
    shortLabel: '512 · 高显存',
    description: '最高网格细节；速度最慢，需要更多显存',
  },
] as const

export type TripoResolution = (typeof TRIPO_RESOLUTION_OPTIONS)[number]['value']

export function getTripoResolutionOption(resolution?: number) {
  return TRIPO_RESOLUTION_OPTIONS.find((option) => option.value === resolution) ?? TRIPO_RESOLUTION_OPTIONS[1]
}
