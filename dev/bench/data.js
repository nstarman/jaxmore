window.BENCHMARK_DATA = {
  "lastUpdate": 1794958072955,
  "repoUrl": "https://github.com/nstarman/jaxmore",
  "entries": {
    "jaxmore Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "nstarman@users.noreply.github.com",
            "name": "Nathaniel Starkman",
            "username": "nstarman"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3d9709a77ac1afbd93bbf746a312b65ddacd9b2a",
          "message": "⚡️ perf(bounded_while_loop): single cond per step to avoid O(carry) copies (#30)\n\n* ⚡️ perf(bounded_while_loop): single cond per step to avoid O(carry) copies\n\nNested lax.cond made XLA copy the carry each scan step. Flatten to one\ncond gated on `not done and cond_fn(val)`. cond_fn is now also evaluated\nafter termination; documented.\n\nFixes #29\n\n* ⚡️ perf(bounded_while_loop): skip cond_fn after termination\n\nEvaluating cond_fn on the frozen carry made early-exit loops with an\nO(n) predicate ~2-3x slower than main. Gate cond_fn behind its own\nbool-returning cond (sequential, not nested), which keeps the carry\nin-place and restores the original semantics.\n\n\nCo-authored-by: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-08T21:40:15-04:00",
          "tree_id": "ef08ff64f72b884c270c6a1cf8037b698fc77a06",
          "url": "https://github.com/nstarman/jaxmore/commit/3d9709a77ac1afbd93bbf746a312b65ddacd9b2a"
        },
        "date": 1791665035515,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 29.61499590041047,
            "unit": "iter/sec",
            "range": "stddev: 0.0006210357407305523",
            "extra": "mean: 33.766676968749465 msec\nrounds: 32"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 98232.90756433607,
            "unit": "iter/sec",
            "range": "stddev: 0.000002509971353758645",
            "extra": "mean: 10.179888031361243 usec\nrounds: 23971"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_large_carry_jit[2000]",
            "value": 35691.56245519256,
            "unit": "iter/sec",
            "range": "stddev: 0.000005037344313343097",
            "extra": "mean: 28.017826377183884 usec\nrounds: 15430"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_large_carry_jit[128000]",
            "value": 21539.430591639913,
            "unit": "iter/sec",
            "range": "stddev: 0.000006424329028859501",
            "extra": "mean: 46.42648261965335 usec\nrounds: 11910"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_early_exit_expensive_cond_jit",
            "value": 2359.210696951797,
            "unit": "iter/sec",
            "range": "stddev: 0.000011793115332626406",
            "extra": "mean: 423.8705772621511 usec\nrounds: 2155"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_closure",
            "value": 782.0415141516094,
            "unit": "iter/sec",
            "range": "stddev: 0.0000992853006726928",
            "extra": "mean: 1.2787044957387472 msec\nrounds: 704"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_carry",
            "value": 774.823507949516,
            "unit": "iter/sec",
            "range": "stddev: 0.00010012086599766657",
            "extra": "mean: 1.290616494905257 msec\nrounds: 687"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_via_step_kw",
            "value": 773.505066717315,
            "unit": "iter/sec",
            "range": "stddev: 0.00008749400683284257",
            "extra": "mean: 1.2928163538008988 msec\nrounds: 684"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_empty_batch_skipping",
            "value": 1196.0487921391434,
            "unit": "iter/sec",
            "range": "stddev: 0.00004481911481075529",
            "extra": "mean: 836.0862922753272 usec\nrounds: 958"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 116623.82352573102,
            "unit": "iter/sec",
            "range": "stddev: 0.000001408213318200068",
            "extra": "mean: 8.574577387092505 usec\nrounds: 51475"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 112592.45503516041,
            "unit": "iter/sec",
            "range": "stddev: 0.000001499949480716642",
            "extra": "mean: 8.881589798248202 usec\nrounds: 54873"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 112679.68204422142,
            "unit": "iter/sec",
            "range": "stddev: 0.000001500943427956279",
            "extra": "mean: 8.874714428174793 usec\nrounds: 55142"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 229968.48048155324,
            "unit": "iter/sec",
            "range": "stddev: 9.484859954777614e-7",
            "extra": "mean: 4.348422000728115 usec\nrounds: 96815"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 119050.3924745483,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013953772099204045",
            "extra": "mean: 8.399804311554783 usec\nrounds: 40774"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 112751.04384513499,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015492882079623972",
            "extra": "mean: 8.869097490339094 usec\nrounds: 56262"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 81141.49615000999,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017091943498766664",
            "extra": "mean: 12.324150372471003 usec\nrounds: 43492"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2076.9465484569287,
            "unit": "iter/sec",
            "range": "stddev: 0.00008072735417836157",
            "extra": "mean: 481.4760402683217 usec\nrounds: 1341"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 1638.2097319833824,
            "unit": "iter/sec",
            "range": "stddev: 0.000026937066820424878",
            "extra": "mean: 610.4224510919604 usec\nrounds: 1053"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 1846.833580658776,
            "unit": "iter/sec",
            "range": "stddev: 0.00003348800644250437",
            "extra": "mean: 541.4673040779853 usec\nrounds: 1128"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nstarman@users.noreply.github.com",
            "name": "Nathaniel Starkman",
            "username": "nstarman"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0048b4058769c4961cb93dd90762c4cb8b5a099f",
          "message": "✨ feat(loop): add `loop` and `loop_strategies` (#35)\n\n* ✨ feat(loop): add `loop` and `loop_strategies`\n\n`jaxmore.loop(strategy, f, signature=...)` maps `f` over the broadcast\nbatch dimensions of its arguments with one of `Vectorize`, `VMap`,\n`Scan`, `NoLoop` or `Determine` (Scan on CPU, Vectorize elsewhere).\nThe strategy tags live in `jaxmore.loop_strategies`.\n\n* ♻️ refactor(loop): infer outputs, accept pytree args and kwargs\n\n- `Vectorize` is now nested `jax.vmap`, so no strategy needs or checks the\n  output part of `signature`, and pytree outputs may differ in core shape.\n- Positional arguments may be pytrees; a signature entry applies to every\n  leaf. Keyword arguments are passed to `f` unmapped.\n- `Determine` picks its strategy when called, not when built.\n\n* 🐛 fix(loop): build `Determine`'s candidates once, satisfy pylint\n\n- `Determine` built its scan/vectorize wrapper on every call; build both\n  once and only choose between them when called.\n- Silence pylint: `_signature` is unused by `NoLoop`, and the public\n  re-export repeats the private `__all__` by design.\n\n* 🐛 fix(loop): index instead of broadcasting, validate signatures, fix NoLoop\n\nFixes from two independent audits of #35:\n\n- `VMap`/`Scan` indexed-per-element instead of pre-broadcasting every\n  argument, so a shared argument is never copied (80 MB -> 68 B/9 KB on\n  the audit's example); empty batches fall back to the nested form.\n- `loop(NoLoop, f)` no longer rewraps `f` onto itself, which set\n  `f.__wrapped__ = f` and broke `inspect` and `jax.jit`.\n- Parse signatures strictly (`ValueError` when malformed), and check that\n  named core dimensions agree and fixed sizes match, which also catches\n  pytree leaves with different core dims.\n- Raise a clear `TypeError` when no positional array is mapped or the\n  strategy is not a strategy class; name the class when a tag is\n  instantiated; return arrays for scalar-batch `Vectorize`.\n- Tests: assert which `Determine` branch ran, cover the tag contract,\n  empty/partially broadcast batches and the new errors. Document the\n  changes and list `loop` in the README features.\n\n* ♻️ refactor(loop): one nested pass for every strategy\n\n`Scan` is now the same nested mapping as `Vectorize`, with `jax.lax.map`\nat each batch level instead of `jax.vmap`, so the flat-index machinery\n(`_element`, `_run_vmap`, `_run_scan`, `_pad`, `_unflatten`) and the\nempty-batch fallbacks go away. `VMap` runs as `Vectorize`; it stays a\nseparate tag for libraries that dispatch on it.\n\nAlso rebuild arguments from one treedef, look strategies up in a dict\nwithout the MRO walk, and drop the redundant strategy type check.\n\nCo-authored-by: Claude Sonnet 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-22T18:45:41-04:00",
          "tree_id": "b25be084548a10161d92792f0cdd8a290fb6a03b",
          "url": "https://github.com/nstarman/jaxmore/commit/0048b4058769c4961cb93dd90762c4cb8b5a099f"
        },
        "date": 1794958072240,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 33.68334564436862,
            "unit": "iter/sec",
            "range": "stddev: 0.000539561252521181",
            "extra": "mean: 29.688262281249543 msec\nrounds: 32"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 101621.42678521755,
            "unit": "iter/sec",
            "range": "stddev: 0.000001940274365170653",
            "extra": "mean: 9.840444398734478 usec\nrounds: 24298"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_large_carry_jit[2000]",
            "value": 37858.233841029265,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035478979472920525",
            "extra": "mean: 26.4143331196882 usec\nrounds: 15604"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_large_carry_jit[128000]",
            "value": 22701.9638498794,
            "unit": "iter/sec",
            "range": "stddev: 0.000004550051236262533",
            "extra": "mean: 44.049052611160434 usec\nrounds: 10302"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_early_exit_expensive_cond_jit",
            "value": 2344.516711846386,
            "unit": "iter/sec",
            "range": "stddev: 0.00001298258745615428",
            "extra": "mean: 426.5271366790413 usec\nrounds: 1873"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_closure",
            "value": 782.4886235644158,
            "unit": "iter/sec",
            "range": "stddev: 0.00012022125403356277",
            "extra": "mean: 1.2779738514852392 msec\nrounds: 707"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_carry",
            "value": 775.3200040223933,
            "unit": "iter/sec",
            "range": "stddev: 0.00011333905434910385",
            "extra": "mean: 1.2897900154929025 msec\nrounds: 710"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_via_step_kw",
            "value": 775.3613861175352,
            "unit": "iter/sec",
            "range": "stddev: 0.00010036279641046496",
            "extra": "mean: 1.2897211776398836 msec\nrounds: 805"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_empty_batch_skipping",
            "value": 1235.463483264462,
            "unit": "iter/sec",
            "range": "stddev: 0.00006093158618152965",
            "extra": "mean: 809.4128345725788 usec\nrounds: 1076"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 138146.95496682212,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015582601997734725",
            "extra": "mean: 7.2386684182808345 usec\nrounds: 51191"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 133328.71181893954,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010754184662033787",
            "extra": "mean: 7.500259969195537 usec\nrounds: 59734"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 131156.27612836703,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011489895091645644",
            "extra": "mean: 7.624492167048618 usec\nrounds: 55024"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 272274.4314706793,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012318596759778355",
            "extra": "mean: 3.6727649915511367 usec\nrounds: 105843"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 140326.72655712697,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010094559013328764",
            "extra": "mean: 7.1262262331253075 usec\nrounds: 61388"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 135348.76660154678,
            "unit": "iter/sec",
            "range": "stddev: 0.000001602441299209829",
            "extra": "mean: 7.388320005485531 usec\nrounds: 58299"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 97782.33233876071,
            "unit": "iter/sec",
            "range": "stddev: 0.00000134533057943261",
            "extra": "mean: 10.226796355558008 usec\nrounds: 45933"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2227.3531952339063,
            "unit": "iter/sec",
            "range": "stddev: 0.00003151160808183123",
            "extra": "mean: 448.96337147597495 usec\nrounds: 1206"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 1777.9326791352182,
            "unit": "iter/sec",
            "range": "stddev: 0.000033363930491743544",
            "extra": "mean: 562.4509925124935 usec\nrounds: 1202"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 1841.8743699613324,
            "unit": "iter/sec",
            "range": "stddev: 0.00002383672005554241",
            "extra": "mean: 542.9251941982307 usec\nrounds: 1241"
          }
        ]
      }
    ]
  }
}