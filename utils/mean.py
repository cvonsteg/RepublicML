def mean(x):
    sum_of_vals = 0
    for i in range(len(x)):
        sum_of_vals += int(x[i])

    return sum_of_vals / len(x)


if __name__ == "__main__":
    x1 = [1, 2, 3, 4, 5]
    x2 = [30, 35, 60, 61]
    print(f'Mean of {x1} is {mean(x1)}')
    print(f'Mean of {x2} is {mean(x2)}') 
