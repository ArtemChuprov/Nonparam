# nonparam — минимальный JAX-прототип AA|CG

Цель этой папки — держать понятный Python-прототип без LAMMPS: построить ту же ASE-систему, что в `boundary_test.py`, проверить boundary unpack/pack и посчитать энергию JAX-MD с cross-вкладом.

## Запуск

```bash
pyenv activate science_env
pip install jax jax-md numpy ase
cd nonparam
python bilayer_cross.py
```

VS Code debug: F5 → **bilayer_cross**.

## Файлы

| File | Role |
|------|------|
| `bilayer_cross.py` | главный запуск: сборка системы, логи, энергия, один JAX-шаг |
| `constants.py` | типы частиц, LJ-параметры, масштаб CG |
| `boundary.py` | геометрия unpack/pack из `boundary_test.py` |
| `structure.py` | ASE bilayer, типизация, box |
| `energy.py` | `scale_wrapper`, homogeneous energy, cross energy |
| `simulation.py` | один простой velocity-Verlet шаг и сохранение соседей |

## Что считается

1. Строится ASE bilayer: fine `Al` + coarse `Cu`, как в `boundary_test.py`.
2. Логируются boundary-сферы:
   - A / unpack: реальные частицы + фантомы `Ag`;
   - B / pack: CG + surviving fine как `Au`.
3. Считается `E_homogeneous`: AA-AA LJ и CG-CG через `scale_wrapper`.
4. Считается `E_cross`: unpack/pack cross-вклад добавляется внутри `energy_fn`.
5. Выполняется один минимальный JAX velocity-Verlet шаг с нулевыми скоростями.

Файлы результата лежат в `out/`:

- `1_full.xyz`
- `2_sphere_A_before.xyz`, `3_sphere_A_after.xyz`
- `4_sphere_B_before.xyz`, `5_sphere_B_after.xyz`
- `neighbors_before.txt`, `neighbors_after.txt`
- `6_after_jax_step.xyz`

## Cutoffs по типам

В `constants.py`:

- `R_CUT_AA = 5` Å — fine (type 1)
- `R_CUT_CG = 10` Å — coarse (type 2), physical
- `NBR_CUT = max(R_CUT_AA, R_CUT_CG)` — радиус neighbor list

В `energy.py` матрица `species_cutoff_matrix()` задаёт cutoff на эффективном расстоянии `B*d`:
AA-AA → 5, CG-CG → `10 * 0.5 = 5` (это physical 10 Å при `B=0.5`).

## scale_wrapper

`scale_wrapper` теперь не привязан к LJ. Это общий декоратор pair-potential:

```python
U_scaled(dr) = A * U_base(B * dr)
```

Поэтому LJ — только частный случай:

```python
lj_pair = with_cutoff(radial_pair(lj_radial))
scaled_lj_pair = scale_wrapper(lj_pair)
```

Если нужен другой pair-potential, он должен иметь jax-md сигнатуру `potential(dr, **params)`. Тогда его можно обернуть тем же `scale_wrapper`.

## Cross / Path 1

Вывод по комментариям другой нейронки: cross-структура должна быть встроена в **одну JAX energy function**, а не выполняться как отдельная подготовка координат перед MD. Поэтому:

- topology unpack/pack строится один раз на `R0`;
- внутри `energy_fn(R, neighbor)` фантомы считаются как `R[center] + offset`;
- pack-связи считаются по текущим координатам `R[CG]` и `R[fine]`;
- значит `jax.grad(energy_fn)` видит cross-вклад.

В коде это место помечено большим комментарием `DEBUG HERE` в `energy.py` и `bilayer_cross.py`.

## Важно

- `boundary.py` не надо менять без сверки с `/boundary_test.py`.
- `test.ipynb` в текущем проекте пустой; полезной логики из него сейчас нет.
- LAMMPS-трек (`lammps/USER-QUASI/`) отдельно, этот прототип его не запускает.
