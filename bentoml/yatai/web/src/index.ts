import { getExpressApp } from './server'

// 0 is node, 1 is index.ts
const args = process.argv.slice(2);
// going to assume channel address at 0 and port at 1 for now
const grpcServerAddress = args[0] || 'localhost:50051';
const port = args[1] || 3000;

const app = getExpressApp(grpcServerAddress);

app.listen(port, () => console.log(`Running at http://localhost:${port}`))
