## HSR Relics OCR
一个识别 崩坏：星穹铁道 中的遗器道具的脚本。 受[Yas](https://github.com/wormtql/yas)启发编写。

### 使用
* 克隆本项目，安装`requirements.txt`中的依赖。
* 将游戏调整为 **中文文本** **窗口化** **1600x900分辨率**，并保证游戏窗口不要最小化，且在置于前台时能够在屏幕中完整展示（可以被其他窗口遮挡）。
* 在游戏中尽量找一个接近黑色的场景，打开背包中的遗器栏，滚动到需要开始扫描的位置。可以只扫描一种遗器，也可以设置筛选条件。
* 在项目根目录执行
```shell
python ./src/extract.py -n 1000
```
**其中的`1000`请替换成自己需要扫描的遗器数量。** 执行时会从前向后扫描，并在完成后忽略扫描到的完全相同的遗器。

**在执行过程中请不要操作电脑。** 如果发生非预期情况，按下 `Win + D` 最小化所有窗口即可使扫描过程停止。

扫描完成的结果会记录在项目根目录下的 `result.yaml` 文件中。

三星及以下的遗器扫描结果可能不准确。建议不要包括在扫描目标中。

游戏帧率低下的情况下可能扫描不准确，如有这样的报告请提交Issue。

### 变更模型
该项目会识别的遗器套装设置在 `config.yaml` 中。如果其中设置的内容不全，可以进行修改后重新训练模型。

训练模型需要用到游戏中的字体文件。一般来说如果电脑上安装有游戏的话会自动寻找字体文件。

在项目根目录执行
```shell
python ./src/model_training.py
```
重新生成数据集并训练模型。训练后的模型存储在 `./model` 中。