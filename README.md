# DPL
The Dick-Cheney-Memorial Programming Language is a BLAZINGLY slow, statically-typed, LLVM-backend shitlang.

The Definitely-Worth-Using Programming Language is a toy project I started as a hobby.

I started by following [this tutorial](https://www.youtube.com/playlist?list=PLCJHRjnsxJFoK8e-RaNZUa7R4BaPqczHX) but made modifications where I wanted to.

## Example
```
fn add(a: i32, b: i32) -> i32 => a + b;

fn double(x: i32) -> i32 => x * 2;

struct Point {
    x: i32;
    y: i32;
}

struct Circle {
    center: Point;
    radius: i32;
}

union OptionI32 {
    Some(i32),
    None
}

fn main() -> i32 {
    let foo: i32 = 10;
    let bar: &i32 = &foo;
    let baz: i32 = *bar;
    printf("baz = %i\n", baz);

    let p: Point = new Point {
        x = 10;
        y = 5;
    };
    let c: Circle = new Circle {
        center = p;
        radius = 10;
    };
    let c2: Circle = new Circle {
        center = new Point {
            x = 100;
            y = p.x |> double;
        };
        radius = 1;
    };
    printf("%i\n", c2.center.x);

    let opt: OptionI32 = OptionI32::Some(10);
    let res: i32 = match opt {
        OptionI32::Some(val) => {
            val
        },
        OptionI32::None => {
            0
        },
    };
    if res == 0 {
        printf("None!\n");
    } else {
        printf("Some (%i)!\n", res);
    };

    0
}
```

## Requirements
It might work on other versions, but here are the versions I developed on:
Python 3.12.7
llvmlite 0.45.1
llvm 21.1.0

Then just pass the file path when running it as an argument :)