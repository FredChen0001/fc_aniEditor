# AniEditor 数据集说明

欢迎使用 AniEditor 开源数据集。本数据集旨在为动画图片编辑与生成相关的研究和开发提供高质量的标注数据。

## 数据集来源

AniEditor 数据集来源于danbooru2024，筛选掉NSFW数据和其它rar、gif等非图片数据，每张图重新保存到jpg格式。保留数据中的tag_string,score,up_score,down_score,fav_count,rating,created_at字段，并用Doubao-Seed-1.6-flash-250715模型对每张图片用中英文进行描述。

## 数据集规模

aniEditor 数据集包含：
- 总计 300k 张图片， 大小100GB

## 数据字段说明

| 数据文件类型 | 描述                                  |
|--------|-------------------------------------|
|  image_width  | 图片的宽度                               |
| image_height  | 图片的长度                               |
| tag_string  | 同原数据集tag_string                     |
| score  | 同原数据集score,得分数(up_score-down_score) |
| up_score   | 同原数据集up_score,点赞数                   |
| down_score   | 同原数据集down_score,                    |
| fav_count   | 同原数据集fav_count,收藏数                  |
| rating   | 同原数据集rating                         |
| created_at   | 同原数据集created_at，图片的上传时间             |
| file   | 原数据集文件名                             |
| subdir   | 原数据集目录名                             |
| caption_en   | 英文描述                                |
| caption_ch   | 中文描述                                |
| file_new   | 图片文件名                               |

## 数据结构

数据集文件组织结构如下：
```
aniEditor/
├── imgs/           # 图片数据
├──file_list.jsonl         # 标注信息JSON文件
└── samples/ #1000张演示数据
```
## 下载地址
百度云盘  https://pan.baidu.com/s/1va0MjZFbxQJaz6n2mkfBMw?pwd=1234

## 引用

如果您在研究中使用了本数据集，请引用以下文献：
```
@misc{AniEditor2025,
  author       = {FredChen0001},
  title        = {{AniEditor: A Comprehensive Dataset for Animation Editing Research}},
  howpublished = {\url{https://github.com/FredChen0001/fc_aniEditor}},
  year         = {2025},
  note         = {GitHub repository},
  publisher    = {GitHub},
}
```