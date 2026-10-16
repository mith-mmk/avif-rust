const fs=require('fs');
const module_=new WebAssembly.Module(fs.readFileSync(process.argv[3]));
const api=require(process.argv[2]);
const cases=api.verify_shared();
if(cases!==5)throw Error('case count mismatch');
process.stdout.write(JSON.stringify({runtime:process.version,cases,all_passed:true,host_imports:WebAssembly.Module.imports(module_),bindings:'wasm-bindgen nodejs; no mocked imports'})+'\n');
