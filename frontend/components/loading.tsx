"use client";

export default function FullScreenLoader() {
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-background/70 backdrop-blur">
      <div className="absolute -top-20 -right-20 h-56 w-56 rounded-full bg-[radial-gradient(circle_at_center,theme(colors.cyan.400/.3),transparent_70%)] animate-pulse" />
      <div className="absolute -bottom-24 -left-10 h-56 w-56 rounded-full bg-[radial-gradient(circle_at_center,theme(colors.blue.500/.25),transparent_70%)] animate-pulse" />
      <div className="relative flex flex-col items-center gap-6">
        <div className="w-16 h-16 border-4 border-cyan-400/40 border-t-cyan-500 rounded-full animate-spin"></div>
        <div className="text-center">
          <h2 className="text-lg font-semibold tracking-tight">StockPulse</h2>
          <p className="text-sm text-foreground/70 mt-1">Loading…</p>
        </div>
      </div>
    </div>
  );
}
