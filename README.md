Тестовое задание
-----------------------------------
Как запустить?

    git clone https://github.com/Firally/TestTaskVK.git

    docker build -t model_name .

Train:

    docker run -e TASK=train -v volume_name:/models model_name

Test:

    docker run -e TASK=test -v volume_name:/models model_name
