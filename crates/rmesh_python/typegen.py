"""
Generate a `.pyi` type stub from the rustdoc JSON output.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# absolute location of current directory and PYI output

cwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))
root = os.path.abspath(os.path.join(cwd, "..", ".."))
target = os.path.join(root, "target")
pyi = os.path.join(cwd, "__init__.pyi")


def run_json_docgen():
    os.system("cargo rustdoc -- --output-format json -Z unstable-options")


def run_formatter():
    os.system(f"ruff format {pyi} && ruff check --fix {pyi} && ruff format {pyi}")


def get_typings():
    run_json_docgen()
    json_path = os.path.join(target, "doc", "rmesh.json")
    with open(json_path, encoding="utf8") as src:
        return json.load(src)["index"]


PRIMITIVES = {
    "f32": "float",
    "f64": "float",
    "i8": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "u8": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
    "bool": "bool",
    "usize": "int",
}

REPLACEMENTS = {"String": "str", "Cow": "bytes"}

TypeInfo = dict[str, Any]
FuncArg = tuple[str, TypeInfo]


def clean_type(typename: str) -> str:
    if typename in REPLACEMENTS:
        return REPLACEMENTS[typename]
    else:
        return typename


def get_inner_type(info: TypeInfo):
    try:
        args = info["resolved_path"]["args"]["angle_bracketed"]["args"]
    except KeyError:
        return None

    return next((a["type"] for a in args if "type" in a), None)


def clean_name(name: str) -> str:
    return name.replace("crate::", "")


def format_type(info: TypeInfo, classname: str | None = None) -> str:
    if classname is not None:
        classname = clean_name(classname)

    if "tuple" in info:
        innertypes = ",".join(format_type(t) for t in info["tuple"])
        return f"tuple[{innertypes}]"
    elif "primitive" in info:
        return PRIMITIVES[info["primitive"]]
    elif "generic" in info:
        if info["generic"] == "Self" and classname is not None:
            return classname  # '"' + classname + '"'
        else:
            return info["generic"]
    elif "resolved_path" in info:
        resolved_name = info["resolved_path"].get(
            "name", info["resolved_path"].get("path")
        )
        if resolved_name == "Option":
            return f"{format_type(get_inner_type(info), classname)} | None"
        elif resolved_name == "Result" or resolved_name == "anyhow::Result":
            return format_type(get_inner_type(info), classname)
        elif resolved_name == "Vec":
            return f"list[{format_type(get_inner_type(info), classname)}]"
        elif resolved_name.startswith("PyReadonlyArray"):
            prim = get_inner_type(info)["primitive"]
            if prim == "f32":
                return "NDArray[float32]"
            elif prim == "f64":
                return "NDArray[float64]"
            elif prim == "i64":
                return "NDArray[int64]"
            else:
                raise ValueError(info)
                return "NDArray"
        else:
            return clean_name(clean_type(resolved_name))  # '"' + resolved_name + '"'
    elif "borrowed_ref" in info:
        reftype = info["borrowed_ref"]["type"]
        return format_type(reftype, classname)
    elif "slice" in info:
        stype = info["slice"]
        assert stype["primitive"] == "u8", f"Weird slice: {info}"
        return "bytes"
    elif "array" in info:
        array_type = info["array"]["type"]
        # TODO: consider if there's any way to nicely type fixed-size arrays?
        return f"list[{format_type(array_type, classname)}]"
    else:
        return "Any"
        # raise ValueError(f"Unknown type: {info}")


def format_arg(name: str, info: TypeInfo, classname: str | None = None) -> str:
    if name == "self":
        return "self"
    return f"{name}: {format_type(info, classname)}"


def remove_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s


@dataclass
class FuncInfo:
    name: str
    args: list[FuncArg]
    ret: TypeInfo | None
    doc: str | None

    def decl(self, classname: str | None = None) -> list[str]:
        ret = []
        arglist = ", ".join(format_arg(arg[0], arg[1], classname) for arg in self.args)
        fname = self.name
        if fname == "new":
            fname = "__init__"
            arglist = "self, " + arglist
        elif fname.startswith("py_get_"):
            ret.append("@property")
            fname = remove_prefix(fname, "py_get_")
        elif fname.startswith("py_"):
            fname = remove_prefix(fname, "py_")

        res = f"{fname}({arglist})"
        if self.ret is not None and fname != "__init__":
            res += f" -> {format_type(self.ret, classname)}"
        ret.append(f"def {res}:")
        if self.doc is not None:
            doclines = remove_prefix(self.doc, "(pyfunc)").strip().split("\n")
            ret.append('    """')
            ret.extend("    " + line.rstrip() for line in doclines)
            ret.append('    """')
        else:
            ret.append('    """ Undocumented function """')
        return ret


def extract_func(item: dict[str, Any]) -> FuncInfo:
    fname = item["name"]
    funcinfo = item["inner"]["function"]
    if "decl" in funcinfo:
        decl = funcinfo["decl"]
    elif "sig" in funcinfo:
        decl = funcinfo["sig"]
    else:
        raise RuntimeError(f"Function {fname} missing 'sig' or 'decl'!")
    args = decl["inputs"]
    retval = None
    if "output" in decl and decl["output"] is not None:
        retval = decl["output"]
    doc = None
    if "docs" in item:
        doc = item["docs"]
    return FuncInfo(fname, args, retval, doc)


def find_methods(typings: dict[str, TypeInfo]) -> dict[str, list[FuncInfo]]:
    res = defaultdict(list)
    for v in typings.values():
        if ("docs" in v) and isinstance(v["docs"], str):
            docstr = v["docs"]
            if docstr.startswith("(pymethods)"):
                impl = v["inner"]["impl"]
                resolve = impl["for"]["resolved_path"]
                classname = resolve.get("name", resolve.get("path"))
                for item in impl["items"]:
                    res[classname].append(extract_func(typings[str(item)]))
            elif docstr.startswith("(pyfunc)"):
                res["__global"].append(extract_func(v))
            else:
                print(f"Unknown docstr: {docstr}")

    return res


def emit_methods(methods: list[FuncInfo], classname: str | None) -> str:
    indent = ""
    res = ""
    if (classname is not None) and (classname != "__global"):
        res = f"class {classname}:\n"
        indent = "    "

    bodies = []
    for meth in methods:
        body = "\n".join([indent + line for line in meth.decl(classname)]) + "\n"
        bodies.append(body)
    return res + "\n".join(sorted(bodies))


# ruff postprocessing will remove anything that wasn't used
PREAMBLE = "\n".join(
    [
        "from typing import Any",
        "from numpy.typing import NDArray",
        "from numpy import float64, float32, int64, uint32",
    ]
)


if __name__ == "__main__":
    run_json_docgen()
    print("Emitted JSON")
    typings = get_typings()
    meths = find_methods(typings)

    print(meths)

    items = [emit_methods(v, k) for k, v in meths.items()]
    print(typings)
    items = sorted(items)

    with open(pyi, "w", encoding="utf8") as dest:
        dest.write(PREAMBLE)
        dest.write("\n\n")
        dest.write("\n".join(items))

    print("Wrote stub file")
    run_formatter()
    print("Formatted.")
