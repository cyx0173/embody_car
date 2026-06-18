# 中文 LaTeX 期末报告模板

这个模板使用 `ctexart`，推荐用 XeLaTeX 编译，适合中文期末报告、课程论文和实验报告。

## 文件结构

- `main.tex`：主文档，包含标题页、摘要、目录、正文、图表、公式、代码和参考文献示例。
- `references.bib`：BibTeX 参考文献示例。
- `figures/`：放置报告图片。
- `Makefile`：使用 `latexmk` 一键编译。

## 编译方式

如果安装了 `latexmk`：

```bash
make
```

或者手动编译：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

编译完成后会生成 `main.pdf`。

## 常用修改位置

1. 在 `main.tex` 顶部修改题目、课程名、姓名、学号和学院/专业。
2. 在 `abstract` 环境中填写摘要和关键词。
3. 从 `引言` 开始替换正文内容。
4. 图片放入 `figures/`，再使用 `\includegraphics` 插入。
5. 参考文献写入 `references.bib`，正文中用 `\cite{文献键}` 引用。
