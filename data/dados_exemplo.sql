INSERT INTO glucose_data (glucose_mg_dl, spectral_transmittance_data, box_temperature_celsius, measurement_phase, real_concentration, volunteer_notes)
VALUES
-- Dados da Fase 1: In Vitro (Soluções controladas)
(98.5, '{"410nm": 0.12, "435nm": 0.15, "560nm": 0.35, "940nm": 0.85}', 25.2, 'in_vitro', 100.0, NULL),
(112.3, '{"410nm": 0.10, "435nm": 0.14, "560nm": 0.30, "940nm": 0.78}', 25.5, 'in_vitro', 110.0, NULL),

-- Dados da Fase 2: In Vivo (Leitura no dedo de voluntários)
(105.1, '{"410nm": 0.08, "435nm": 0.11, "560nm": 0.25, "940nm": 0.70}', 32.4, 'in_vivo', 105.0, 'Voluntário 1: Pele clara, espessura normal'),
(140.8, '{"410nm": 0.09, "435nm": 0.12, "560nm": 0.28, "940nm": 0.65}', 33.1, 'in_vivo', 142.0, 'Voluntário 2: Pele parda, leitura após refeição');