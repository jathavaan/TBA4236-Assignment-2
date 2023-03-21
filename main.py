import numpy as np


def read_file() -> list:
    with open("data/data.txt", "r") as f:
        data = f.readlines()

    data_list = []
    for line in data:
        d = line.split(" ")
        name, N, E, h, H, method = d

        data_dict = {
            "name": name,
            "N": float(N),
            "E": float(E),
            "h": float(h),
            "H": float(H),
            "method": method
        }

        data_list.append(data_dict)

    return data_list


def N(h: float, H: float) -> float:
    return h - H


def LSM(data: list) -> tuple:
    table = []
    weight_matrix = np.eye(len(data))
    parameters = np.array(["A", "B", "C", "D", "E", "F"])
    N = np.array([d["h"] - d["H"] for d in data])

    for i in range(len(data)):
        d = data[i]

        name, y, x, h, H, method = d.values()
        row = [x ** 2, x * y, y ** 2, x, y, 1]
        table.append(row)

        weight_matrix[i][i] = 4 if method == "Niv" else 1

    A = np.array(table)
    A_trans = A.T

    normal_equation = np.dot(np.dot(A_trans, weight_matrix), A)
    sol = np.linalg.solve(normal_equation, np.dot(np.dot(A_trans, weight_matrix), N))

    std_dev = np.sqrt(np.diag(np.linalg.inv(np.dot(np.dot(A.T, weight_matrix), A))))

    for i in range(len(sol)):
        print(f"{parameters[i]} = {sol[i]} +- {std_dev[i]}")

    A, B, C, D, E, F = sol
    return A, B, C, D, E, F


def main() -> None:
    data = read_file()
    A, B, C, D, E, F = LSM(data)


if __name__ == "__main__":
    main()
