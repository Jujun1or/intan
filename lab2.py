import math
import sys
from dataclasses import dataclass


def down(value):
    return math.nextafter(value, -math.inf)


def up(value):
    return math.nextafter(value, math.inf)


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError('Левая граница интервала превышает правую')

    @staticmethod
    def of(value):
        return value if isinstance(value, Interval) else Interval(value, value)

    def __add__(self, other):
        other = self.of(other)
        return Interval(down(self.lo + other.lo), up(self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-self.of(other))

    def __rsub__(self, other):
        return self.of(other) - self

    def __mul__(self, other):
        other = self.of(other)
        values = [a * b for a in (self.lo, self.hi)
                  for b in (other.lo, other.hi)]
        return Interval(down(min(values)), up(max(values)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.of(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError('Делитель содержит нуль')
        return self * Interval(down(1 / other.hi), up(1 / other.lo))

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError('Степень должна быть целой и неотрицательной')
        result = Interval(1, 1)
        for _ in range(power):
            result = result * self
        return result

    @property
    def mid(self):
        return (self.lo + self.hi) / 2

    @property
    def rad(self):
        return (self.hi - self.lo) / 2

    @property
    def mag(self):
        return max(abs(self.lo), abs(self.hi))

    def intersect(self, other):
        left, right = max(self.lo, other.lo), min(self.hi, other.hi)
        if left > right:
            raise ValueError('Пересечение внешних оценок оказалось пустым')
        return Interval(left, right)

    def hull(self, other):
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))


def square(x):
    values = [x.lo * x.lo, x.hi * x.hi]
    return Interval(0 if x.lo <= 0 <= x.hi else down(min(values)), up(max(values)))


def exponential(x):
    return Interval(down(math.exp(x.lo)), up(math.exp(x.hi)))


def sine(x):
    if x.hi - x.lo >= 2 * math.pi:
        return Interval(-1, 1)
    values = [math.sin(x.lo), math.sin(x.hi)]
    first = math.ceil((x.lo - math.pi / 2) / math.pi)
    last = math.floor((x.hi - math.pi / 2) / math.pi)
    values.extend(1.0 if k % 2 == 0 else -1.0 for k in range(first, last + 1))
    return Interval(max(-1, down(min(values))), min(1, up(max(values))))


def cosine(x):
    values = [math.cos(x.lo), math.cos(x.hi)]
    first = math.ceil(x.lo / math.pi)
    last = math.floor(x.hi / math.pi)
    values.extend(1.0 if k % 2 == 0 else -1.0 for k in range(first, last + 1))
    if x.hi - x.lo >= 2 * math.pi:
        return Interval(-1, 1)
    return Interval(max(-1, down(min(values))), min(1, up(max(values))))


def exprel(t):
    return 1.0 if t == 0 else math.expm1(t) / t


def exprel_interval(x):
    return Interval(down(exprel(x.lo)), up(exprel(x.hi)))


def bisect_root(function, left, right):
    fl, fr = function(left), function(right)
    if fl == 0:
        return left
    if fr == 0:
        return right
    if fl * fr > 0:
        raise ValueError('На концах отрезка нет смены знака')
    for _ in range(100):
        middle = (left + right) / 2
        if middle == left or middle == right:
            return middle
        fm = function(middle)
        if fm == 0:
            return middle
        if fl * fm <= 0:
            right = middle
        else:
            left, fl = middle, fm
    return (left + right) / 2


def f1(x):
    return x ** 3 - 3 * x ** 2 + 2


def f2(x):
    return math.expm1(x) - 2 * x


def f3(x, y):
    return x * x + y * y + math.sin(x) + y * math.cos(x)


def natural1(x):
    return x ** 3 - 3 * x ** 2 + 2


def horner1(x):
    return x * (x * (x - 3)) + 2


def natural2(x):
    return exponential(x) - 2 * x - 1


def equivalent2(x):
    return exponential(x) - 2 * (x + 0.5)


def natural3(x, y):
    return x ** 2 + y ** 2 + sine(x) + y * cosine(x)


def equivalent3(x, y):
    c = cosine(x)
    return square(x) + square(y + c / 2) + sine(x) - square(c) / 4


def mean_value(function, box, center, derivatives):
    result = Interval.of(function(*center))
    for x, c, d in zip(box, center, derivatives):
        result = result + d * (x - c)
    return result


def baumann(box, derivatives):
    ratios = [max(-1, min(1, d.mid / d.rad)) if d.rad else 0
              for d in derivatives]
    left = tuple(x.mid - p * x.rad for x, p in zip(box, ratios))
    right = tuple(x.mid + p * x.rad for x, p in zip(box, ratios))
    return left, right


def bicentered(function, box, derivatives):
    left, right = baumann(box, derivatives)
    a = mean_value(function, box, left, derivatives)
    b = mean_value(function, box, right, derivatives)
    return a.intersect(b), (left, right), (a, b)


def slope1(x, c):
    def value(t):
        return t * t + (c - 3) * t + c * c - 3 * c
    points = [x.lo, x.hi]
    critical = (3 - c) / 2
    if x.lo <= critical <= x.hi:
        points.append(critical)
    values = [value(t) for t in points]
    return Interval(down(min(values)), up(max(values)))


def slope2(x, c):
    return math.exp(c) * exprel_interval(x - c) - 2


def hausdorff(a, b):
    return max(abs(a.lo - b.lo), abs(a.hi - b.hi))


def calculate():
    x1, x2 = Interval(0, 3), Interval(0, 2)
    x, y = Interval(-1, 2), Interval(-1, 1)
    exact1 = Interval(-2, 2)
    exact2 = Interval(1 - 2 * math.log(2), math.exp(2) - 5)
    root = bisect_root(lambda t: 2*t + math.cos(t) + math.cos(t)*math.sin(t)/2, -1, 2)
    minimum_point = (root, -math.cos(root)/2)
    corners = [(a, b) for a in (-1, 2) for b in (-1, 1)]
    maximum_point = max(corners, key=lambda p: f3(*p))
    exact3 = Interval(f3(*minimum_point), f3(*maximum_point))
    d1 = Interval(-3, 9)
    d2 = exponential(x2) - 2
    dx = Interval(down(-2 + math.cos(1) - math.sin(1)),
                  up(4 + math.cos(2) + math.sin(2)))
    dy = Interval(down(-2 + math.cos(2)), 3)
    rows = []
    bics, centers, components, slopes = [], [], [], []
    for function, box, derivative, natural, equiv, slope in [
            (f1, x1, d1, natural1, horner1, slope1),
            (f2, x2, d2, natural2, equivalent2, slope2)]:
        table = [('B.1, исходная запись', natural(box)),
                 ('B.2, эквивалентная запись', equiv(box))]
        current_slopes = []
        for c in (box.lo, box.mid, box.hi):
            table.append((f'B.3, центр {c:g}', mean_value(function, (box,), (c,), (derivative,))))
        for c in (box.lo, box.mid, box.hi):
            si = slope(box, c)
            current_slopes.append((c, si))
            table.append((f'B.4, центр {c:g}', function(c) + si * (box-c)))
        bic, cs, parts = bicentered(function, (box,), (derivative,))
        table.append(('B.5, два центра', bic))
        rows.append(table)
        bics.append(bic)
        centers.append(cs)
        components.append(parts)
        slopes.append(current_slopes)
    bic3, cs3, parts3 = bicentered(f3, (x, y), (dx, dy))
    splitx = (natural3(Interval(-1, 0.5), y), natural3(Interval(0.5, 2), y))
    splity = (natural3(x, Interval(-1, 0)), natural3(x, Interval(0, 1)))
    rows3 = [('E.3, исходная запись', natural3(x, y)),
             ('E.4, выделение квадратов', equivalent3(x, y)),
             ('E.5, середина (0.5; 0)', mean_value(f3, (x,y), (0.5,0), (dx,dy))),
             ('E.5, центр (0; 0)', mean_value(f3, (x,y), (0,0), (dx,dy))),
             ('E.6, два центра', bic3),
             ('E.7, дробление по первой координате', splitx[0].hull(splitx[1])),
             ('E.7, дробление по второй координате', splity[0].hull(splity[1]))]
    rows.append(rows3)
    centers.append(cs3)
    components.append(parts3)
    lipschitz = [
        ('Первая функция, исходная запись', d1.mag, 45, rows[0][0][1].rad, 45*x1.rad),
        ('Первая функция, схема Горнера', d1.mag, 27, rows[0][1][1].rad, 27*x1.rad),
        ('Вторая функция, исходная запись', d2.mag, math.exp(2)+2,
         rows[1][0][1].rad, (math.exp(2)+2)*x2.rad),
        ('Вторая функция, эквивалентная запись', d2.mag, math.exp(2)+2,
         rows[1][1][1].rad, (math.exp(2)+2)*x2.rad)]
    result = dict(rows=rows, exact=[exact1,exact2,exact3], slopes=slopes,
                  derivatives=[d1,d2,(dx,dy)], centers=centers, components=components,
                  minimum_point=minimum_point, maximum_point=maximum_point,
                  splitx=splitx, splity=splity, lipschitz=lipschitz,
                  coordinate_bound=dx.mag*x.rad+dy.mag*y.rad,
                  expression_bound=6*x.rad+3*y.rad)
    verify(result)
    return result


def verify(result):
    tolerance = 2e-10
    for table, exact in zip(result['rows'], result['exact']):
        for name, estimate in table:
            if estimate.lo > exact.lo+tolerance or estimate.hi < exact.hi-tolerance:
                raise AssertionError('Внешняя оценка не содержит найденную область значений: '+name)
    for row in result['lipschitz']:
        if row[3] > row[4]+tolerance:
            raise AssertionError('Не выполнена оценка полуширины по дереву выражения')
    for index in (0,1):
        for j in range(3):
            mv = result['rows'][index][2+j][1]
            sl = result['rows'][index][5+j][1]
            if sl.lo < mv.lo-tolerance or sl.hi > mv.hi+tolerance:
                raise AssertionError('Наклонная оценка не вложена в среднезначную')
    if result['rows'][2][0][1].rad > result['coordinate_bound']+tolerance:
        raise AssertionError('Не выполнено проверяемое покоординатное неравенство')
    for i in range(101):
        x = -1+3*i/100
        for j in range(61):
            y = -1+2*j/60
            value = f3(x,y)
            exact = result['exact'][2]
            if not exact.lo-tolerance <= value <= exact.hi+tolerance:
                raise AssertionError('Контрольная сетка противоречит найденным экстремумам')


def number(value):
    if abs(value) < 1e-12:
        return '0'
    return f'{value:.8f}'.rstrip('0').rstrip('.')


def show(interval):
    return '['+number(interval.lo)+'; '+number(interval.hi)+']'


def main():
    arguments = sys.argv[1:]
    if arguments in (['-h'], ['--help']):
        print('Лабораторная работа №2. Вариант 23.')
        print('Запуск: python lab2.py')
        print('Параметр --latex: вывести строки таблиц в формате LaTeX.')
        return
    if arguments not in ([], ['--latex']):
        raise ValueError('Неизвестные параметры. Для справки используйте --help')
    latex = arguments == ['--latex']
    result = calculate()
    for index, (table, exact) in enumerate(zip(result['rows'], result['exact']),1):
        print(f'Функция №{index}. Область значений: {show(exact)}')
        for name, estimate in table:
            distance = hausdorff(estimate,exact)
            if latex:
                print(f'{name} & ${show(estimate)}$ & ${distance:.6f}$ '+r'\\')
            else:
                print(f'{name}: {show(estimate)}; расстояние: {distance:.8f}')
        for label, center, part in zip(('Первый','Второй'),result['centers'][index-1],result['components'][index-1]):
            print(f'{label} центр Бауманна: '+str(tuple(round(v,10) for v in center))+'; оценка: '+show(part))
        print()
    print('Интервалы производных:')
    print('Первая функция: '+show(result['derivatives'][0]))
    print('Вторая функция: '+show(result['derivatives'][1]))
    print('Частные производные: '+', '.join(show(d) for d in result['derivatives'][2]))
    for i, values in enumerate(result['slopes'],1):
        for center, slope in values:
            print(f'Функция №{i}, центр {center:g}, интервал наклонов: {show(slope)}')
    print('Минимум третьей функции в точке: '+str(result['minimum_point']))
    print('Максимум третьей функции в точке: '+str(result['maximum_point']))
    print('Разбиение по первой координате: '+', '.join(show(s) for s in result['splitx']))
    print('Разбиение по второй координате: '+', '.join(show(s) for s in result['splity']))
    print('Оценки Липшица:')
    for name, exact_l, expression_l, radius, bound in result['lipschitz']:
        print(f'{name}: константа функции {exact_l:.8f}; константа выражения {expression_l:.8f}; полуширина {radius:.8f} <= {bound:.8f}')
    print(f"Третья функция: полуширина {result['rows'][2][0][1].rad:.8f}; покоординатная граница {result['coordinate_bound']:.8f}; граница по дереву выражения {result['expression_bound']:.8f}")
    print('Все проверки пройдены. Вычисления выполнены в двойной точности.')


if __name__ == '__main__':
    main()
