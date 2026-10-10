window.BENCHMARK_DATA = {
  "lastUpdate": 1791665035823,
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
      }
    ]
  }
}