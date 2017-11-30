import tkinter as tk

# import PhotoScan


input_field = False
label = False


def trans_chunck_init():
    top = tk.Tk()
    global input_field
    global label
    print('init Transform Chunck')

    input_field = tk.Text(top, width=60, height=10)
    input_field.pack()
    defaluttext = '1.000000 0.000000 0.000000 0.000000\n0.000000 1.000000 0.000000 0.000000\n0.000000 0.000000 1.000000 0.000000\n0.000000 0.000000 0.000000 1.000000\n'

    input_field.insert("1.0", defaluttext)
    transform_chunk_btn = tk.Button(top,
                                    text='transform chunck!',
                                    command=transform_chunck)

    transform_chunk_btn.pack()

    label = tk.Label(top)

    label.pack()
    top.mainloop()


def transform_chunck():
    text = input_field.get("1.0", tk.END)
    try:
        # text = '1.000000 0.000000 0.000000 0.000000\n0.000000 1.000000 0.000000 0.000000\n0.000000 0.000000 1.000000 0.000000\n0.000000 0.000000 0.000000 1.000000\n'
        lines = text.splitlines()
        lines = list(filter(lambda x: len(x) > 2, lines))

        print("Do Transform!")
        matrix_list = []
        if len(lines) != 4:
            print("unvalid number of rows")
            raise

        for line in lines:
            line_value = line.split()
            if len(line_value) != 4:
                print("unvalid number of columns", line_value)
                raise

            line_value = list(map(lambda x: float(x), line_value))
            matrix_list.append(line_value)

        trafo_matrix = PhotoScan.Matrix(matrix_list)
        PhotoScan.app.document.chunk.transform.matrix = trafo_matrix
        print(PhotoScan.app.document.chunk.transform.matrix)

    except:
        label.config(text='The Matrix is not valid.\n'
                          'Please use a 4x4 Matrix with blanks as seperator')


        # trans_chunck_init()
