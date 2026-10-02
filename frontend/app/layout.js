import './globals.css';

export const metadata = {
  title: 'MOTANAXY — Local Code Lab',
  description: 'A local playground for the MOTANAXY experimental code model.',
};

export default function RootLayout({ children }) {
  return <html lang="th"><body>{children}</body></html>;
}
