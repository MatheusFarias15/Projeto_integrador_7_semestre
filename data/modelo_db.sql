CREATE TABLE glucose_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    glucose_mg_dl NUMERIC,
    spectral_transmittance_data JSONB,
    box_temperature_celsius NUMERIC,
    measurement_phase VARCHAR(50) CHECK (measurement_phase IN ('in_vitro', 'in_vivo')),
    real_concentration NUMERIC,
    volunteer_notes TEXT
);

-- Opcional: Criar um índice na data para acelerar consultas futuras no aplicativo
CREATE INDEX idx_glucose_created_at ON glucose_data(created_at DESC);