

if __name__ == '__main__':

    filename = "./zonotope.txt"
    with open(filename, "w") as text_file:
        print(f"100 ", file=text_file)
        print(f"2 ", file=text_file)
        for i in range(100):
            print(f"0 0.1 ", file=text_file)

    print("generating zonotope")
