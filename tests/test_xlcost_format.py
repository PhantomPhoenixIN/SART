"""Unit tests for converting the pinned XLCoST token rows to source text."""

import ast
import unittest

from src.data.xlcost_format import detokenize_xlcost


class XlcostFormatTests(unittest.TestCase):
    def test_python_indentation_and_multichar_operators(self) -> None:
        row = "def f ( x ) : NEW_LINE INDENT x += 1 NEW_LINE return x NEW_LINE DEDENT"
        result = detokenize_xlcost(row, "Python")
        self.assertEqual(result, "def f (x) :\n    x += 1\n    return x\n")
        ast.parse(result)

    def test_python_string_markers(self) -> None:
        row = "print ( \" Hello ▁ world STRNEWLINE next TABSYMBOL line \" )"
        result = detokenize_xlcost(row, "Python")
        self.assertIn(r"Hello world\nnext\tline", result)
        ast.parse(result)

    def test_java_operators_and_package_dots(self) -> None:
        row = "java . util . Arrays . sort ( a ) ; NEW_LINE x ++ ;"
        self.assertEqual(
            detokenize_xlcost(row, "Java"), "java.util.Arrays.sort (a);\nx++;\n"
        )

    def test_cpp_include_and_operator_reconstruction(self) -> None:
        row = "# include < bits / stdc + + . h > NEW_LINE x - > value ++ ;"
        self.assertEqual(
            detokenize_xlcost(row, "C++"), "#include <bits/stdc++.h>\nx->value++;\n"
        )

    def test_unknown_language_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detokenize_xlcost("int x ;", "Rust")


if __name__ == "__main__":
    unittest.main()
