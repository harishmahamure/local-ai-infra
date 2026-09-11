import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { StoreProvider } from "./api/store";
import { AppShell } from "./layout/AppShell";
import { Downloads } from "./pages/Downloads";
import { ImageOp } from "./pages/ImageOp";
import { Images } from "./pages/Images";
import { JobDetail } from "./pages/JobDetail";
import { Jobs } from "./pages/Jobs";
import { Models } from "./pages/Models";
import { Overview } from "./pages/Overview";
import { Runtime } from "./pages/Runtime";
import { ShotFlow } from "./pages/ShotFlow";
import { Shots } from "./pages/Shots";
import { WallpaperOp } from "./pages/WallpaperOp";
import { Wallpapers } from "./pages/Wallpapers";

export default function App() {
  return (
    <StoreProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Overview />} />
            <Route path="runtime" element={<Runtime />} />
            <Route path="images" element={<Images />} />
            <Route path="images/:op" element={<ImageOp />} />
            <Route path="wallpapers" element={<Wallpapers />} />
            <Route path="wallpapers/:op" element={<WallpaperOp />} />
            <Route path="shots" element={<Shots />} />
            <Route path="shots/:op" element={<ShotFlow />} />
            <Route path="jobs" element={<Jobs />} />
            <Route path="jobs/:id" element={<JobDetail />} />
            <Route path="models" element={<Models />} />
            <Route path="downloads" element={<Downloads />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </StoreProvider>
  );
}
