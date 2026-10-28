from fractions import Fraction as Q
from itertools import combinations, permutations, product
from pathlib import Path
import json


def matrix(rows):
    return [[Q(str(value)) for value in row] for row in rows]


RECT = matrix([['0.95', 1], ['1.05', 1], ['1.10', 1]])
SQUARE = matrix([['1.10', '0.90', '1.10'],
                 ['1.40', '1.00', '0.80'], ['0.80', '1.40', '1.20']])


def mv(a, x):
    return [sum((v * w for v, w in zip(row, x)), Q(0)) for row in a]


def reduce_rows(a):
    rows = [list(row) for row in a]
    pivot_row = 0
    for col in range(len(rows[0])):
        pivot = next((i for i in range(pivot_row, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][col]
        rows[pivot_row] = [v / scale for v in rows[pivot_row]]
        for i in range(len(rows)):
            if i != pivot_row:
                scale = rows[i][col]
                rows[i] = [v - scale * w for v, w in zip(rows[i], rows[pivot_row])]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return rows


def rank(a):
    return sum(any(row) for row in reduce_rows(a))


def inverse(a):
    n = len(a)
    if any(len(row) != n for row in a) or rank(a) != n:
        raise ValueError('Требуется квадратная невырожденная матрица')
    augmented = [row + [Q(i == j) for j in range(n)] for i, row in enumerate(a)]
    return [row[n:] for row in reduce_rows(augmented)]


def determinant(a):
    if len(a) == 1:
        return a[0][0]
    return sum(((-1) ** j * a[0][j] * determinant(
        [row[:j] + row[j + 1:] for row in a[1:]])
        for j in range(len(a))), Q(0))


def solve_equations(equations):
    n = len(equations)
    augmented = [[Q(v) for v in coefficients] + [Q(rhs)]
                 for coefficients, rhs in equations]
    if rank([row[:n] for row in augmented]) != n:
        return None
    return tuple(row[-1] for row in reduce_rows(augmented))


def feasible_vertices(dimension, equalities, inequalities):
    vertices = set()
    for active in combinations(inequalities, dimension - len(equalities)):
        point = solve_equations(list(equalities) + list(active))
        if point is not None and all(
                sum(c * x for c, x in zip(coefficients, point)) <= rhs
                for coefficients, rhs in inequalities):
            vertices.add(point)
    return sorted(vertices, key=lambda point: (point[-1], point))


def uniform_vertices(center):
    n = len(center[0])
    equality = [(tuple([Q(1)] * n + [Q(0)]), Q(1))]
    candidates = []
    for signs in product((-1, 1), repeat=n):
        bounds = []
        for j in range(n + 1):
            row = [Q(0)] * (n + 1)
            row[j] = Q(-1)
            bounds.append((tuple(row), Q(0)))
        for row in center:
            signed = [value * sign for value, sign in zip(row, signs)]
            bounds += [(tuple(signed + [Q(-1)]), Q(0)),
                       (tuple([-v for v in signed] + [Q(-1)]), Q(0))]
        for point in feasible_vertices(n + 1, equality, bounds):
            x = tuple(sign * y for sign, y in zip(signs, point[:-1]))
            candidates.append((point[-1], x))
    if not candidates:
        raise RuntimeError('Допустимые вершины не найдены')
    return sorted(candidates)


def first_column_vertices(center):
    if any(len(row) != 2 or row[1] != 1 for row in center):
        raise ValueError('Второй столбец должен состоять из единиц')
    bounds = [((Q(0), Q(-1)), Q(0))]
    for value, _ in center:
        bounds.extend([((Q(-1), Q(-1)), -value),
                       ((Q(1), Q(-1)), value)])
    vertices = feasible_vertices(2, [], bounds)
    if not vertices:
        raise RuntimeError('Допустимые вершины не найдены')
    return vertices


def construct(a, mask, x):
    residual = mv(a, x)
    result = []
    for i, row in enumerate(a):
        capacity = sum(mask[i][j] * abs(x[j]) for j in range(len(x)))
        if not capacity:
            if residual[i]:
                raise ValueError('Невозможно обнулить невязку допустимыми изменениями')
            result.append(list(row))
            continue
        result.append([value - residual[i] * mask[i][j]
                       * ((x[j] > 0) - (x[j] < 0)) / capacity
                       for j, value in enumerate(row)])
    return result


def verify(a, mask, delta, b, x):
    inside = all(abs(b[i][j] - a[i][j]) <= delta * mask[i][j]
                 for i in range(len(a)) for j in range(len(x)))
    residual = mv(b, x)
    actual_rank = rank(b)
    minors = [determinant([b[i] for i in indices])
              for indices in combinations(range(len(b)), len(x))]
    if not (any(x) and inside and not any(residual)
            and actual_rank == len(x) - 1 and not any(minors)):
        raise AssertionError('Точная проверка матрицы не пройдена')
    return dict(inside=inside, residual=residual, rank=actual_rank,
                maximal_minors=minors,
                max_deviation=max(abs(v - w) for ar, br in zip(a, b)
                                  for v, w in zip(ar, br)))


def main():
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    all_rect = uniform_vertices(RECT)
    one_column = first_column_vertices(RECT)
    all_square = uniform_vertices(SQUARE)
    d1, x1 = all_rect[0]
    t, dc = one_column[0]
    xc = (Q(1), -t)
    d2, x2 = all_square[0]
    cases = [('rect_all', RECT, matrix([[1, 1]] * 3), d1, x1, len(all_rect)),
             ('rect_column', RECT, matrix([[1, 0]] * 3), dc, xc, len(one_column)),
             ('square_all', SQUARE, matrix([[1] * 3] * 3), d2, x2, len(all_square))]
    records, console = {}, []
    titles = {
        'rect_all': 'А1: изменяются оба столбца',
        'rect_column': 'А1: изменяется только первый столбец',
        'square_all': 'А2: квадратная матрица',
    }
    for name, center, mask, delta, x, count in cases:
        witness = construct(center, mask, x)
        check = verify(center, mask, delta, witness, x)
        records[name] = dict(delta=delta, kernel=x, witness=witness,
                             vertex_count=count, **check)
        console += [titles[name], f'Количество опорных решений: {count}',
                    f'Минимальный радиус: {delta} = {float(delta):.12f}',
                    'Вектор ядра: (' + ', '.join(map(str, x)) + ')',
                    f'Диапазон особенности: от {delta} включительно до плюс бесконечности',
                    'Точечная матрица:']
        console += ['  ' + '  '.join(map(str, row)) for row in witness]
        console += [
            'Принадлежность интервальной матрице: подтверждена' if check['inside']
            else 'Принадлежность интервальной матрице: не подтверждена',
            'Невязка: (' + ', '.join(map(str, check['residual'])) + ')',
            f"Ранг: {check['rank']}",
            'Максимальные миноры: ' + ', '.join(map(str, check['maximal_minors'])),
            f"Максимальное отклонение от центра: {check['max_deviation']}", '']
    inv = inverse(SQUARE)
    cube = [(signs, sum(map(abs, mv(inv, signs))))
            for signs in product((1, -1), repeat=3) if signs[0] == 1]
    if 1 / max(norm for _, norm in cube) != d2:
        raise AssertionError('Проверка через норму обратной матрицы не пройдена')
    radius = Q(1, 20)
    determinants = [determinant([
        [SQUARE[i][j] + radius * signs[3*i+j] for j in range(3)]
        for i in range(3)]) for signs in product((-1, 1), repeat=9)]
    lower = upper = Q(0)
    for permutation in permutations(range(3)):
        parity = (-1) ** sum(permutation[i] > permutation[j]
                            for i in range(3) for j in range(i+1, 3))
        lo = hi = Q(1)
        for i, j in enumerate(permutation):
            products = [a*b for a in (lo, hi)
                        for b in (SQUARE[i][j]-radius, SQUARE[i][j]+radius)]
            lo, hi = min(products), max(products)
        lower, upper = ((lower+lo, upper+hi) if parity == 1
                        else (lower-hi, upper-lo))
    records['inverse'] = inv
    records['cube_vertices'] = [dict(signs=s, norm=n) for s, n in cube]
    records['determinant_comparison'] = dict(
        delta=radius, exact=[min(determinants), max(determinants)],
        natural_interval=[lower, upper])
    console += ['Проверка через норму обратной матрицы:',
                *(f'Вершина {s}: норма равна {n}' for s, n in cube),
                f'При радиусе {radius} точный диапазон определителя: '
                f'[{min(determinants)}, {max(determinants)}]',
                f'Естественное интервальное расширение определителя: [{lower}, {upper}]']
    (out / 'results.json').write_text(json.dumps(records, default=str, indent=2), encoding='utf-8')
    (out / 'console.txt').write_text('\n'.join(console) + '\n', encoding='utf-8')
    print('\n'.join(console))


if __name__ == '__main__':
    main()
