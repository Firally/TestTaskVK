Тестовое задание
-----------------------------------

Данный проект содержит Jupyter Notebook с решением, а также решение, обернутое в docker.

Как запустить?

    git clone https://github.com/Firally/TestTaskVK.git

Сборка:

    docker build -t model_name .

Train:

    docker run -e TASK=train -v volume_name:/models model_name

Test:

    docker run -e TASK=test -v volume_name:/models model_name
