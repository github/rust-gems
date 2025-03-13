# Contributing

## Building the WASM/JS package

The code for the wasm + js wrapper package is stored in the `js` directory. To build it:

```sh
cd js
npm i
npm run build
```

The npm package will be output to `js/pkg`.

To run a quick sanity check of the JS package:

```sh
npm test
```

To publish the package to npm, run:

```sh
cd js/pkg
npm publish
```
