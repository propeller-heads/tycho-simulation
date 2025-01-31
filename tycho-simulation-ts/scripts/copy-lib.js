const fs = require('fs');
const path = require('path');
const os = require('os');

// Determine platform-specific library extension
const platform = os.platform();
let libExt;
switch (platform) {
  case 'darwin':
    libExt = '.dylib';
    break;
  case 'linux':
    libExt = '.so';
    break;
  case 'win32':
    libExt = '.dll';
    break;
  default:
    throw new Error(`Unsupported platform: ${platform}`);
}

// Source and destination paths
const projectRoot = path.resolve(__dirname, '..', '..');
const targetDir = path.join(projectRoot, 'target', 'release');
const libName = `libtycho_simulation_ts${libExt}`;
const sourcePath = path.join(targetDir, libName);
const destPath = path.join(projectRoot, 'tycho-simulation-ts', 'index.node');

// Check if source exists
if (!fs.existsSync(sourcePath)) {
  console.error(`Library not found at ${sourcePath}`);
  console.log('Contents of release directory:');
  try {
    const files = fs.readdirSync(targetDir);
    console.log(files);
  } catch (err) {
    console.error('Could not read release directory:', err);
  }
  process.exit(1);
}

// Copy the file
try {
  fs.copyFileSync(sourcePath, destPath);
  console.log(`Successfully copied ${sourcePath} to ${destPath}`);
} catch (err) {
  console.error('Error copying library:', err);
  process.exit(1);
} 