# Translator

Простой скрипт для автоматизации перевода датасетов, на данный момент реализова перевод для:

- [isaiahbjork/chain-of-thought-sharegpt](https://huggingface.co/datasets/isaiahbjork/chain-of-thought-sharegpt)
- [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

Возможные варианты модели:

- [utrobinmv/t5_translate_en_ru_zh_small_1024](https://huggingface.co/utrobinmv/t5_translate_en_ru_zh_small_1024) -
  small модель (наиболее быстрая)
- [utrobinmv/t5_translate_en_ru_zh_large_1024](https://huggingface.co/utrobinmv/t5_translate_en_ru_zh_large_1024) -
  large модель (самый качественный перевод)
- [utrobinmv/t5_translate_en_ru_zh_base_200](https://huggingface.co/utrobinmv/t5_translate_en_ru_zh_base_200) - base
  модель (не уступающая по качеству модели large), но для более коротких текстов и более быстрая.

Полезные ссылки:

- [Сравнение локальных моделей машинного перевода для английского, китайского и русского языков](https://habr.com/ru/articles/791522/)
- [New argos model en_ru for add argospm-index](https://community.libretranslate.com/t/new-argos-model-en-ru-for-add-argospm-index/311)
- https://github.com/EvilFreelancer/impruver
