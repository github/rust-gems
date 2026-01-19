# Sizing your HRIBLT

Because the HRIBLT is rateless, it is possible to append additional data in order to make it decoding possible. That is, it does not need to be sized in advance like a standard invertible bloom lookup table.

Regardless, there are some advantages to getting the size of your decoding session correct the first time. An example might be if you're performing set reconciliation over some RPC and you want to minimise the number of round trips it takes to perform a decode.

## Coded Symbol Multiplier

The number of coded symbols required to find the difference between two sets is proportional to the difference between the two sets. The following chart shows the relationship between the number of coded symbols required to decode HRIBLT and the size of the diff. Note that the size of the base set (before diffs were added) was fixed.

`y = len(coded_symbols) / diff_size`

![Coded symbol multiplier](../evaulation/overhead/overhead.png)

For small diffs, the number of coded symbols required per value is larger, after a difference of approximately 100 values the coefficient settles on around 1.3 to 1.4.

You can use this chart, combined with an estimate of the diff size (perhaps from a `geo_filter`) to increase the probability that you will have a successful decode after a single round-trip while also minimising the amount of data sent.
