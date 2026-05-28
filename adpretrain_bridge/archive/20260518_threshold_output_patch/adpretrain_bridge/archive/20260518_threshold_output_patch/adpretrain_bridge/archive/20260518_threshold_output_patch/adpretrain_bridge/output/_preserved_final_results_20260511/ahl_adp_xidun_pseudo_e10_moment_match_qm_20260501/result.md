# Result Status

状态：优化实验已重提。

## 已完成

- 新增解耦脚本：`/ghome/huangjd/code/detected/adpretrain_bridge/moment_match_cache.py`
- 已生成独立缓存根：`/gdata1/huangjd/data/xidun_mvtec_format_adpretrain_xidun_pseudo_e10_moment_match_qm_20260501`
- 缓存检查通过：train 560、test 339，`feature=(512,14,14)`，`feature_scale=(512,7,7)`
- 已补 no-augmentation alias：`aug -> feature`，`aug_scale -> feature_scale`
- PBS 文件已确认结尾两个空行
- launcher 已修正为无 BOM UTF-8，首字节为 `23 21 2f 75`

## 作业记录

- `202571.Ghead`：失败。原因是 AHL no-augmentation 路径仍访问 `aug/aug_scale`，优化缓存缺少 alias。
- `202572.Ghead`：失败。原因是 launcher 带 UTF-8 BOM，Docker 报 `exec format error`。
- `202573.Ghead`：当前有效重提，等待/运行中。

## 待完成

- AHL 30 epoch
- `metrics/ahl_aligned_metrics_full.md`
- 与 plain ADPretrain 替换、DRA baseline 做对比