#!/usr/bin/env node
// Preflight checks for `npm run dev` at the repo root. Fails loudly and
// exits non-zero before concurrently ever starts a process, instead of
// letting the frontend/backend fail silently or half-start.

import { existsSync, readFileSync } from "node:fs";
import { createServer } from "node:net";
import { execSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const RESET = "\x1b[0m";

const errors = [];
const warnings = [];

function checkCommand(cmd, hint) {
    try {
        execSync(`command -v ${cmd}`, { stdio: "ignore" });
    } catch {
        errors.push(`'${cmd}' is not on PATH. ${hint}`);
    }
}

function portFree(port) {
    return new Promise((resolve) => {
        const srv = createServer();
        srv.once("error", () => resolve(false));
        srv.once("listening", () => srv.close(() => resolve(true)));
        srv.listen(port, "127.0.0.1");
    });
}

checkCommand("uv", "Install it (https://docs.astral.sh/uv/) — the backend is run via `uv run`.");

const envPath = path.join(REPO_ROOT, ".env");
if (!existsSync(envPath)) {
    errors.push(
        ".env is missing at the repo root. Copy .env.example to .env and fill in the CDSE credentials."
    );
} else {
    const envText = readFileSync(envPath, "utf8");
    const env = Object.fromEntries(
        envText
            .split("\n")
            .map((line) => line.trim())
            .filter((line) => line && !line.startsWith("#") && line.includes("="))
            .map((line) => {
                const idx = line.indexOf("=");
                return [line.slice(0, idx).trim(), line.slice(idx + 1).trim()];
            })
    );

    for (const key of ["CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"]) {
        if (!env[key] && !process.env[key]) {
            errors.push(
                `${key} is not set in .env. The Scene Input CDSE search/preview routes need it — the backend will start but every search request will fail with 503.`
            );
        }
    }
    if (!env.OPENTOPOGRAPHY_API_KEY && !process.env.OPENTOPOGRAPHY_API_KEY) {
        warnings.push(
            "OPENTOPOGRAPHY_API_KEY is not set in .env — only needed for the offline DEM download scripts, not for serving the app, so this is not blocking."
        );
    }
}

if (!existsSync(path.join(REPO_ROOT, "frontend", "node_modules"))) {
    errors.push("frontend/node_modules is missing. Run `npm install --prefix frontend` first.");
}
if (!existsSync(path.join(REPO_ROOT, "node_modules"))) {
    errors.push("Root node_modules is missing. Run `npm install` at the repo root first.");
}

const [backendPortFree, frontendPortFree] = await Promise.all([portFree(8000), portFree(5173)]);
if (!backendPortFree) {
    errors.push("Port 8000 is already in use (backend/uvicorn needs it). Stop whatever is bound to it first, e.g. `lsof -i :8000`.");
}
if (!frontendPortFree) {
    errors.push("Port 5173 is already in use (frontend/vite needs it). Stop whatever is bound to it first, e.g. `lsof -i :5173`.");
}

if (warnings.length) {
    console.warn(`${YELLOW}dev preflight warnings:${RESET}`);
    for (const w of warnings) console.warn(`${YELLOW}  - ${w}${RESET}`);
}

if (errors.length) {
    console.error(`${RED}dev preflight FAILED — not starting anything:${RESET}`);
    for (const e of errors) console.error(`${RED}  - ${e}${RESET}`);
    process.exit(1);
}

console.log("dev preflight OK — starting backend + frontend.");
