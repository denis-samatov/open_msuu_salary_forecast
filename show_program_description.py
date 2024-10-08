import streamlit as st

def show_program_description():
    st.title("Добро пожаловать в Приложение для Прогнозирования Зарплаты")

    st.header("Как пользоваться этим приложением")
    st.markdown("""
    1. **Перейти к Прогнозированию Зарплаты:** Используйте боковую панель для доступа к странице "Прогноз зарплаты".
    2. **Введите необходимые данные:** Укажите необходимую информацию, такую как название должности, местоположение и другие релевантные поля.
    3. **Получить Вакансии и Прогноз:** Нажмите кнопку "Получить вакансии", чтобы собрать и обработать данные. Посмотрите полные прогнозы зарплаты и подробные анализы.
    4. **Анализируйте свою вакансию:** Заполните форму с деталями вашей вакансии и нажмите кнопку "Векторизовать вакансию и спрогнозировать зарплату". Приложение покажет диапазон зарплат для указанной должности.

    **Примечание:** Для оптимальных результатов убедитесь, что предоставляемая информация точна и актуальна.
    """)

    st.header("Как это работает")
    st.markdown("""
    Это приложение использует данные из многочисленных объявлений о вакансиях на HeadHunter (HH) для прогнозирования возможных зарплат для различных ролей. Вот краткий обзор процесса:

    1. **Сбор данных:** Приложение получает данные о вакансиях с HeadHunter (HH) с помощью API.
    2. **Выбор параметров:** Пользователи могут выбрать конкретные регионы, которые их интересуют, выбрать должность, профессиональную роль, чтобы увидеть локализованные данные по вакансиям.
    3. **Обработка данных:** Собранные данные проходят тщательную очистку и предобработку для извлечения ключевой информации. Кроме этого происходит векторизация вакансий.
    4. **Анализ:** Приложение проводит несколько анализов, включая обнаружение аномалий, распределение заработной платы, анализ ключевых навыков и профессиональных ролей.
    5. **Прогнозирование:** С помощью передовых алгоритмов машинного обучения приложение предсказывает возможные зарплаты для указанных ролей, предоставляя как статистические данные, так и анализ трендов.

    """)

    st.header("Базовая аналитика")
    st.markdown("""
    После получения вакансий приложение проводит следующие анализы, чтобы предоставить пользователям углубленные инсайты:

    - **Анализ зарплат:** Предлагает подробные данные о распределении зарплат, включая такие метрики, как медиана, среднее, минимум, максимум.
    - **Анализ навыков:** Определяет и выделяет топ-10 навыков, требуемых для вакансий, помогая пользователям понять ключевые компетенции, востребованные на рынке труда.
    - **Анализ профессиональных ролей:** Анализирует распределение различных профессиональных ролей среди вакансий, давая пользователям обзор сегментации рынка труда.
    """)

    st.header("О приложении")
    st.markdown("""
    Приложение для прогнозирования зарплаты создано, чтобы помогать соискателям и работодателям, предоставляя подробные инсайты по ожиданиям по зарплатам на основе текущих рыночных трендов. Оно использует передовые методы обработки данных и машинного обучения для точного и полезного прогнозирования зарплат.
    """)

    if st.button('Связаться с нами'):
        st.header("Контакты")
        st.markdown("""
        Для любых вопросов или поддержки, пожалуйста, свяжитесь с нами по следующим контактам:
        - **Электронная почта:** [support@salaryforecasting.com](mailto:support@salaryforecasting.com)
        - **Телефон:** +1 (123) 456-7890
        """)

    if st.button('Отказ от ответственности'):
        st.header("Отказ от ответственности")
        st.markdown("""
        Прогнозы зарплат, предоставляемые этим приложением, основаны на исторических данных и текущих рыночных трендах. Они предназначены для использования в качестве ориентира и не должны рассматриваться как точные предсказания. Фактические зарплаты могут варьироваться в зависимости от таких факторов, как индивидуальный опыт, образование и конкретные требования к работе. Пользователям рекомендуется использовать предоставленную информацию в качестве справочной и проводить собственные исследования для точных ожиданий по зарплате.
        """)
