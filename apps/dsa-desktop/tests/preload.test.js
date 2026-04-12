const assert = require('node:assert/strict');
const test = require('node:test');
const Module = require('node:module');

test('preload exposes desktop version from package.json', (t) => {
  const originalLoad = Module._load;
  const exposeInMainWorldCalls = [];

  Module._load = function patchedLoad(request, parent, isMain) {
    if (request === 'electron') {
      return {
        contextBridge: {
          exposeInMainWorld: (...args) => {
            exposeInMainWorldCalls.push(args);
          },
        },
      };
    }
    return originalLoad.call(this, request, parent, isMain);
  };

  const preloadPath = require.resolve('../preload.js');
  delete require.cache[preloadPath];

  t.after(() => {
    Module._load = originalLoad;
    delete require.cache[preloadPath];
  });

  require('../preload.js');

  assert.equal(exposeInMainWorldCalls.length, 1);
  assert.deepEqual(exposeInMainWorldCalls[0], [
    'dsaDesktop',
    {
      version: require('../package.json').version,
    },
  ]);
});
