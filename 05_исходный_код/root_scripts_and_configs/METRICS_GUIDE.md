# 📊 Метрики для сравнения ИИ в симуляции "Лягушка и мухи"

## 🎯 Обзор

Этот документ описывает систему метрик для сравнения трёх архитектур ИИ:
- **ANN** (Artificial Neural Network) — классическая нейросеть с непрерывным обучением
- **SNN** (Spiking Neural Network) — спайковая сеть с LIF-нейронами
- **BioFrog** — биоморфный мозг с полной нейробиологической достоверностью

---

## 📐 Категории метрик

### 1. Task Performance (Результативность)

| Метрика | Формула | Интерпретация |
|---------|---------|---------------|
| **Catch Rate (CR)** | `(Catches / Steps) × 100%` | Процент шагов с поимкой. Выше = лучше |
| **Time-to-Capture (TTC)** | `Steps / Catches` | Среднее число шагов между поимками. Ниже = лучше |
| **Flies per Energy** | `Catches / TotalEnergySpent` | **КЛЮЧЕВАЯ МЕТРИКА**: метаболическая эффективность |

```python
# Пример вывода:
📊 TASK PERFORMANCE:
   Catch Rate:           4.00% (Catches per 100 steps)
   Avg Time-to-Capture:  25.0 steps (Average steps between catches)
   Flies/Energy:         0.005 ⭐ (Key metric: metabolic efficiency)
```

---

### 2. Strategy & Mechanics (Тактика)

| Метрика | Описание | Норма |
|---------|----------|-------|
| **Tongue Success Rate** | % успешных выстрелов языком | 75-92% (зависит от config) |
| **Avg Shot Distance** | Среднее расстояние до мухи при выстреле | 20-30px оптимально |
| **Movement Efficiency** | `TotalDistance / Catches` | Ниже = лучше позиционирование |
| **Avg Alignment** | Скалярное произведение velocity и направления на муху | 0-1, выше = лучше |

```python
🎯 STRATEGY & TACTICS:
   Tongue Success Rate:  77.1% (% successful tongue shots)
   Avg Shot Distance:    23.7px (range: 10.0 - 39.9)
   Movement Efficiency:  6589.02px/catch (lower = better positioning)
   Avg Alignment:        0.679 (velocity alignment with prey)
```

---

### 3. Bio-Plausibility (Биологическая достоверность)

| Метрика | Для кого | Описание |
|---------|----------|----------|
| **Spikes per Catch** | SNN, BioFrog | Нейральная эффективность |
| **Sparsity Ratio** | SNN, BioFrog | Доля "тихих" шагов (биосистемы стремятся к разреженности) |
| **Burst Ratio** | SNN, BioFrog | Доля шагов с ≥2 спайками (салиентные события) |
| **Decision Stability** | Все | Консистентность решений в одинаковых состояниях |

```python
🧬 BIOLOGICAL PLAUSIBILITY:
   Total Spikes:         1999
   Spikes/Catch:         25.30 (fewer = more efficient coding)
   Sparsity Ratio:       0.409 (fraction of silent steps)
   Burst Ratio:          0.290 (≥2 spikes = salient events)
   Decision Stability:   0.977 (1.0 = perfectly stable)
```

---

### 4. Error Cost (Цена ошибки)

| Метрика | Формула | Зачем |
|---------|---------|-------|
| **Avg Recovery Time** | Среднее шагов после промаха до следующей поимки | Скорость восстановления |
| **Opportunity Cost** | Оценка мух, упущенных во время восстановления | Цена неудачи |

```python
⚠️  ERROR COST:
   Avg Recovery Time:    1.1 steps (faster recovery = better)
   Opportunity Cost:     ~18361 flies (missed during recovery)
```

---

### 5. Survival Metrics (Выживаемость)

| Метрика | Порог | Интерпретация |
|---------|-------|---------------|
| **Energy Balance** | `EnergyGain - EnergySpent` | Положительный = устойчиво |
| **Starvation Risk** | MinEnergy ≤ 6.0 | 1.0 = был критический уровень |
| **Survival Status** | Energy > 0 | ALIVE ✅ / STARVED ❌ |

```python
❤️ SURVIVAL STATUS:
   Status:               STARVED ❌
   Avg Energy:           -94.36/30.0
   Energy Range:         -264.37 - 34.19
   Net Energy Balance:   -16291.98 (gain - spent)
   Starvation Risk:      1.0 (experienced critical levels)
```

---

### 6. Developmental Metrics (Онтогенез — только BioFrog)

| Метрика | Описание |
|---------|----------|
| **Time to Competence** | Шагов до достижения положительного баланса энергии |
| **Developmental Cost** | Общее число juvenile-шагов (инвестиция в развитие) |
| **Skill Retention** | Поимок на 100 шагов взрослой жизни |

```python
🌱 DEVELOPMENTAL ONTOGENY:
   Time to Competence:   14 steps (to positive energy balance)
   Developmental Cost:   2000 steps (investment in development)
   Skill Retention:      2.367 (catches per 100 adult steps)
```

---

## 🔧 Использование

### Базовый пример

```python
from metrics_framework import MetricsCollector

# Создание коллектора
collector = MetricsCollector()
collector.initial_energy = 30.0

# В цикле симуляции
for step in range(max_steps):
    agent_state = agent.update(dt, flies)
    
    # Запись шага
    collector.record_step(agent_state, dt=0.01)
    
    # При выстреле языком
    if tongue_shot:
        collector.record_tongue_shot(distance, success)

# Отчёт
collector.print_summary("MyAgent")
```

### Сравнение архитектур

```python
from metrics_framework import compare_architectures

collectors = {
    "ANN": ann_collector,
    "SNN": snn_collector,
    "BioFrog": bio_collector,
}

print(compare_architectures(list(collectors.items())))
```

### Экспорт данных

```python
# Получить все метрики как словарь
metrics_dict = collector.to_dict()

# Доступ к конкретным метрикам
catch_rate = metrics_dict["task_performance_catch_rate"]
energy_balance = metrics_dict["survival_energy_balance"]
sparsity = metrics_dict["bio_plausibility_sparsity_ratio"]
```

---

## 📈 Пример вывода сравнения

```
====================================================================================================
🏆 ARCHITECTURE COMPARISON SUMMARY
====================================================================================================
Architecture    |  Catches |  Catch Rate |   Flies/E |  Sparsity | Energy Bal |   Status
----------------------------------------------------------------------------------------------------
ANN             |       80 |       4.00% |     0.005 |     1.000 |  -16291.98 | ❌ STARVED
SNN             |       79 |       3.95% |     0.005 |     0.409 |  -16664.69 | ❌ STARVED
BioFrog         |       47 |       2.35% |     0.003 |     0.506 |  -16377.87 | ❌ STARVED
====================================================================================================

Legend:
  • Catch Rate: % of steps resulting in catch (higher = better)
  • Flies/E: Flies caught per energy unit (KEY METRIC - metabolic efficiency)
  • Sparsity: Fraction of silent neural steps (bio-realism)
  • Energy Bal: Net energy gain/loss (positive = sustainable)
```

---

## 🎓 Интерпретация результатов

### Ключевые вопросы для исследования:

1. **Эффективность**: Какая архитектура ловит больше мух на единицу энергии?
   - Смотрите `flies_per_energy` — это главная метрика эффективности

2. **Биореализм vs Производительность**: Оправдана ли сложность BioFrog?
   - Сравните `catch_rate` и `flies_per_energy` между архитектурами
   - Оцените `sparsity_ratio` и `decision_stability` для биодостоверности

3. **Цена развития**: Стоит ли "детство" BioFrog своих затрат?
   - Посмотрите `time_to_competence` и `developmental_cost`
   - Сравните с производительностью ANN/SNN, которые не требуют развития

4. **Тактическое мастерство**: Кто лучше позиционируется?
   - `movement_efficiency` — меньше путь на поимку = лучше
   - `avg_shot_distance` — оптимальная дистанция выстрела

5. **Устойчивость**: Кто выживает дольше?
   - `energy_balance` > 0 означает устойчивую стратегию
   - `starvation_risk` показывает, насколько близко к голоду

---

## 📁 Файлы

- `metrics_framework.py` — основной модуль с `MetricsCollector`
- `demo_metrics.py` — демонстрационный скрипт с примерами использования
- `METRICS_GUIDE.md` — этот документ

---

## 💡 Советы по использованию

1. **Запускайте несколько эпизодов**: Один запуск может быть случайным. Усредняйте по 10-30 эпизодам.

2. **Фиксируйте seed**: Для воспроизводимости устанавливайте `random.seed()` перед каждым запуском.

3. **Следите за энергией**: Если все агенты умирают (`STARVED`), возможно, параметры среды слишком жёсткие.

4. **Обращайте внимание на выбросы**: Отдельные метрики могут быть аномальными из-за случайности. Смотрите на тренды.

5. **Используйте визуализацию**: Графики истории энергии, поимок и нейронной активности дают больше инсайтов, чем усреднённые числа.

---

*Документация создана для исследовательского проекта по сравнению ИИ-архитектур в симуляции охоты на мух.*
