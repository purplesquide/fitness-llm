"""
Generate a comprehensive fitness & nutrition Q&A dataset from the
Angela Harter Alger textbook (Nutrition and Physical Fitness, CC BY 4.0).
Covers all 9 chapters: intro, nutrition standards, energy balance,
digestion, carbohydrates, lipids, proteins, metabolism, weight/body image.
Outputs: artifacts/datasets/nutrition_textbook_data.jsonl
"""

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert fitness and nutrition coach built into a workout planning app. "
    "Give evidence-based, concise advice in 3 to 5 sentences. "
    "Always reference the user's level, goal, and current plan when relevant. "
    "Only mention urgent medical escalation for genuine red-flag scenarios. "
    "Use a direct, practical coaching tone."
)

QA_PAIRS = [

    # ─── CHAPTER 1: Introduction to Nutrition & Fitness ─────────────────────

    {
        "user": "What is the difference between unconscious eating and intentional eating?",
        "assistant": (
            "Unconscious eating means choosing food based on convenience, taste, or habit "
            "without considering how it fuels your body — snacking mindlessly while watching TV "
            "is a common example. Intentional eating means listening to hunger and fullness cues "
            "and choosing foods that provide the nutrients your body needs. Neither is all-or-nothing: "
            "the goal is to eat intentionally most of the time while leaving room for occasional "
            "choices you enjoy. Shifting from unconscious to intentional eating is a habit change "
            "that takes time, knowledge, and patience — not a quick fix."
        ),
    },
    {
        "user": "What are the six classes of essential nutrients and what do they do?",
        "assistant": (
            "The six essential nutrients are carbohydrates, lipids (fats), proteins, water, vitamins, "
            "and minerals. Carbohydrates, lipids, and proteins are macronutrients that provide energy: "
            "carbs and protein each yield 4 kcal/g, fat yields 9 kcal/g. Vitamins and minerals are "
            "micronutrients needed in small amounts — they do not provide calories but regulate chemical "
            "reactions that produce energy and maintain body structure. Water makes up about 60% of the "
            "body and is essential for transporting nutrients, regulating temperature, and enabling every "
            "metabolic reaction."
        ),
    },
    {
        "user": "How many calories per gram do carbs, protein, and fat provide?",
        "assistant": (
            "Carbohydrates provide 4 kcal per gram, protein provides 4 kcal per gram, and fat provides "
            "9 kcal per gram — making fat more than twice as calorie-dense as the other two macronutrients. "
            "Alcohol, while not a nutrient, provides 7 kcal per gram with no nutritional value. "
            "Understanding these values helps you estimate the caloric contribution of any food by "
            "reading its macronutrient breakdown on the nutrition label."
        ),
    },
    {
        "user": "What are phytochemicals and why should I eat them?",
        "assistant": (
            "Phytochemicals are naturally occurring compounds in plant foods that give them their colors, "
            "smells, and flavors — they are not considered essential nutrients because no deficiency disease "
            "results from avoiding them. However, they have powerful antioxidant and anti-inflammatory "
            "properties that reduce risk of chronic diseases like cancer and heart disease. The best "
            "strategy is to eat a variety of colorful plants — different colors signal different "
            "phytochemicals — rather than targeting any single source."
        ),
    },
    {
        "user": "What is the difference between structured and unstructured physical activity?",
        "assistant": (
            "Structured activity is planned exercise done with the intent of improving or maintaining "
            "fitness — a gym session, a run, or a swim workout. Unstructured activity, or activities "
            "of daily living, is movement that happens as part of daily life: cooking, walking to your "
            "car, taking stairs. Both burn calories and contribute to health. Research shows that reducing "
            "sedentary time through unstructured activity is independently protective against chronic "
            "disease — even daily gym-goers who sit for 8+ hours have elevated mortality risk."
        ),
    },
    {
        "user": "What are the five basic training principles I should know?",
        "assistant": (
            "The five core training principles are: (1) Progressive overload — you must consistently "
            "increase frequency, duration, or intensity to keep improving; (2) Individuality — no single "
            "program works for everyone, you must account for your body's response; (3) Specificity — "
            "train in ways that match your actual goal (marathon training differs from powerlifting); "
            "(4) Recovery — intense training requires adequate rest for adaptation to occur; "
            "(5) Reversibility — fitness gains reverse when training stops, so consistency is essential."
        ),
    },
    {
        "user": "How much cardio does the ACSM recommend per week?",
        "assistant": (
            "ACSM recommends at least 150 minutes of moderate-intensity cardio per week, which can be "
            "met by doing 30–60 minutes of moderate effort 5 days a week, or 20–60 minutes of vigorous "
            "effort 3 days a week. People unable to meet these minimums still benefit from any amount "
            "of movement — some activity is always better than none. Gradual progression of duration, "
            "frequency, and intensity reduces injury risk and improves long-term adherence."
        ),
    },
    {
        "user": "What are the ACSM strength training recommendations?",
        "assistant": (
            "ACSM recommends training each major muscle group 2–3 days per week using a variety of "
            "exercises and equipment. Two to four sets per exercise with 8–12 reps improve strength "
            "and power for most adults; 15–20 reps develop muscular endurance. At least 48 hours "
            "should pass between resistance sessions targeting the same muscle group to allow for "
            "recovery and adaptation. Beginners and older adults should start with very light to light "
            "intensity before progressing load."
        ),
    },
    {
        "user": "Why is sitting all day dangerous even if I exercise?",
        "assistant": (
            "Prolonged, uninterrupted sitting is independently associated with increased cardiovascular "
            "disease mortality and all-cause mortality — even when accounting for regular exercise. "
            "The longer you sit without breaks, the greater the risk. Breaking up sedentary time every "
            "hour with brief walks or standing counteracts some of this risk. Practically: if you have "
            "a desk job, set a reminder every 60 minutes to stand and move for 5 minutes — it compounds "
            "over the course of a day into significant health benefit."
        ),
    },

    # ─── CHAPTER 2: Nutrition Standards & Guidelines ─────────────────────────

    {
        "user": "What is the difference between an RDA and a UL?",
        "assistant": (
            "The RDA (Recommended Dietary Allowance) is the daily intake level sufficient to meet the "
            "nutrient needs of 97–98% of healthy individuals — it is the target to prevent deficiency. "
            "The UL (Tolerable Upper Intake Level) is the highest daily intake unlikely to cause harm — "
            "staying below it prevents toxicity. If your intake is between the RDA and the UL for a "
            "given nutrient, you do not need to worry about either deficiency or toxicity. Exceeding "
            "the UL is primarily a risk when using high-dose supplements, not from food alone."
        ),
    },
    {
        "user": "What are the key themes of the 2020-2025 Dietary Guidelines for Americans?",
        "assistant": (
            "The theme is 'Make Every Bite Count': prioritize nutrient-dense foods, limit empty-calorie "
            "foods high in saturated fat, added sugar, and sodium, and stay within your calorie needs. "
            "The four key concepts are: follow a healthy pattern at every life stage; customize choices "
            "to personal preferences and budget; focus on food-group needs within calorie limits; and "
            "limit added sugars (under 10% of calories), saturated fat, sodium, and alcohol. Eighty-five "
            "percent of your diet should come from nutrient-dense foods."
        ),
    },
    {
        "user": "What is the difference between nutrient dense, calorie dense, and empty calorie foods?",
        "assistant": (
            "Nutrient-dense foods provide a high ratio of vitamins, minerals, and protein relative to "
            "their calorie content — think vegetables, lean meats, whole grains, and legumes. "
            "Calorie-dense foods are high in calories per serving, but are not necessarily unhealthy — "
            "avocados, olive oil, and nuts qualify. Empty-calorie foods provide calories from added "
            "sugar, sodium, and solid fats with minimal micronutrients — processed snacks and "
            "sugar-sweetened beverages are the main examples. The goal is to fill most of your diet "
            "with nutrient-dense options, leaving room for calorie-dense whole foods."
        ),
    },
    {
        "user": "How do I read a nutrition facts label quickly?",
        "assistant": (
            "Start with serving size — all values on the label are based on one serving, and many "
            "packages contain multiple servings. Then check total calories, followed by saturated fat, "
            "sodium, and added sugars, which you want to minimize. Look for dietary fiber, protein, "
            "vitamin D, calcium, iron, and potassium as nutrients to prioritize. The % Daily Value "
            "column is useful: 5% or less means low in that nutrient, 20% or more means high. "
            "Reading the ingredients list shows you what the food is actually made of — items are "
            "listed from highest to lowest quantity."
        ),
    },
    {
        "user": "Are food structure/function claims approved by the FDA?",
        "assistant": (
            "No — structure/function claims like 'calcium builds strong bones' or 'fiber maintains "
            "bowel regularity' are not pre-approved by the FDA and do not require scientific review "
            "before appearing on packaging. They must include the disclaimer: 'This product is not "
            "intended to diagnose, treat, cure, or prevent any disease.' Health claims that link a "
            "specific food to reduced disease risk (like 'reduces heart disease') do require FDA "
            "approval and must be substantiated by scientific evidence."
        ),
    },

    # ─── CHAPTER 3: Energy Balance ────────────────────────────────────────────

    {
        "user": "What is the difference between BMR and RMR?",
        "assistant": (
            "Basal Metabolic Rate (BMR) is the absolute minimum energy your body needs to sustain "
            "life — measured under highly controlled conditions immediately after waking from deep "
            "sleep, after a 12-hour fast. Resting Metabolic Rate (RMR) is measured after a 12-hour "
            "fast while the subject is awake and at rest — it is approximately 10% higher than BMR "
            "because it includes a bit more activity. For practical purposes, RMR is used because "
            "it is easier to measure, and it represents roughly 60–70% of your total daily calorie "
            "expenditure."
        ),
    },
    {
        "user": "How do I calculate my total daily energy expenditure?",
        "assistant": (
            "First estimate your RMR using the simplified equation: 1 kcal per kg per hour for men, "
            "0.9 kcal per kg per hour for women — multiply by 24 to get daily RMR. Then multiply by "
            "an activity factor: 1.2 for sedentary, 1.3 for somewhat active (1–2 days/week exercise), "
            "1.4 for average (3 days/week), 1.5 for above average (4–5 days/week), 1.6 for very active "
            "(6 days/week intense), 1.7+ for physical labor or professional athletes. For example, a "
            "180 lb (82 kg) man who trains 4 days/week: RMR = 1968 kcal/day × 1.5 = ~2950 kcal/day TEE."
        ),
    },
    {
        "user": "What factors can I change to increase my resting metabolism?",
        "assistant": (
            "The two most impactful modifiable factors are fat-free (lean) mass and avoiding very low "
            "calorie diets. Increasing muscle mass through strength training is the single best way to "
            "raise your resting metabolism permanently — muscle tissue burns more calories at rest than "
            "fat tissue. Conversely, starvation-level calorie restriction can reduce RMR by 20% or more "
            "and these effects can persist long after normal eating resumes. Physiological stress, growth, "
            "and stimulants like caffeine have smaller, temporary effects. Focus on building and "
            "maintaining muscle throughout your life."
        ),
    },
    {
        "user": "How do I create a safe calorie deficit to lose fat without losing muscle?",
        "assistant": (
            "Aim for a 300–500 kcal deficit per day — this produces 0.5–1 lb of fat loss per week "
            "without triggering the metabolic adaptation that accompanies starvation diets. Never drop "
            "below your estimated RMR, as eating below basal needs causes the body to break down muscle "
            "for energy and slows metabolism. Combine moderate caloric restriction with resistance "
            "training to preserve lean mass and keep your RMR elevated. Increase protein intake toward "
            "the higher end of recommendations (1.6–2.2 g/kg) to further protect muscle during a deficit."
        ),
    },
    {
        "user": "How many extra calories do I need to build muscle?",
        "assistant": (
            "A caloric surplus of 300–500 kcal above your estimated total energy expenditure is the "
            "evidence-based range for lean bulking. Larger surpluses do not produce more muscle — excess "
            "calories beyond what muscle synthesis can utilize are stored as fat. Track your weight "
            "weekly: target 0.25–0.5 kg gain per week for intermediate lifters. If you have been training "
            "consistently and your weight is not increasing, you are almost certainly under-eating — "
            "inadequate calories, not inadequate training, is the most common reason for stalled muscle gain."
        ),
    },
    {
        "user": "What is the thermic effect of food?",
        "assistant": (
            "The thermic effect of food (TEF) is the energy your body expends to digest, absorb, and "
            "transport nutrients — it accounts for roughly 10% of total daily calorie expenditure. "
            "If you eat 2000 kcal/day, approximately 200 kcal go toward processing that food. "
            "Protein has the highest TEF (20–30%), followed by carbohydrates (5–10%) and fats (0–3%). "
            "TEF is the smallest component of energy expenditure compared to RMR and physical activity, "
            "so trying to 'boost metabolism' with meal frequency is largely ineffective for meaningful "
            "fat loss."
        ),
    },

    # ─── CHAPTER 4: Digestion & Absorption ───────────────────────────────────

    {
        "user": "What is the difference between mechanical and chemical digestion?",
        "assistant": (
            "Mechanical digestion is purely physical — chewing, stomach churning, and intestinal "
            "segmentation break food into smaller pieces without changing its chemical structure. "
            "Chemical digestion uses enzymes, acids, and bile to break the molecular bonds in food, "
            "converting complex carbohydrates into simple sugars, proteins into amino acids, and fats "
            "into fatty acids and glycerol. Both processes work simultaneously starting in the mouth: "
            "chewing is mechanical, while salivary amylase initiates chemical starch breakdown at the "
            "same time. Most chemical digestion is completed in the small intestine."
        ),
    },
    {
        "user": "Where does most nutrient absorption happen in the body?",
        "assistant": (
            "The small intestine is where almost all nutrient absorption occurs. Its enormous surface "
            "area — roughly 200 m², larger than 100 times your skin — is created by folds, finger-like "
            "villi, and even smaller microvilli on each absorptive cell. Water-soluble nutrients like "
            "amino acids, glucose, and water-soluble vitamins enter capillaries and travel via the "
            "hepatic portal vein directly to the liver. Fat-soluble nutrients — dietary fats, vitamins "
            "A, D, E, and K — are packaged into chylomicrons that enter the lymphatic system and bypass "
            "the liver, depositing directly into the bloodstream."
        ),
    },
    {
        "user": "What is the gut microbiome and how does it affect my fitness?",
        "assistant": (
            "The gut microbiome is the community of microorganisms living in your large intestine — "
            "it functions almost like an organ, influencing metabolism, immune function, recovery, and "
            "even mental health. Athletes tend to have more diverse microbiomes than sedentary people, "
            "and research suggests gut bacteria can influence aerobic capacity and strength. Bacteria "
            "ferment soluble fiber into short-chain fatty acids that reduce gut inflammation and may "
            "serve as an additional fuel during aerobic exercise. The best evidence-based way to "
            "maintain a healthy microbiome is to eat a diverse diet with plenty of minimally processed "
            "plant foods high in fiber."
        ),
    },
    {
        "user": "What is GERD and how should I manage it around training?",
        "assistant": (
            "GERD (gastroesophageal reflux disease) occurs when the lower esophageal sphincter does not "
            "close properly, allowing stomach acid to back up into the esophagus. Around training, avoid "
            "large meals within 2 hours of exercise, especially high-fat and spicy foods which worsen "
            "reflux. Stay upright for at least 3 hours after eating. During exercise, high-impact "
            "activities like running can aggravate symptoms more than cycling or swimming. If symptoms "
            "are frequent or severe, consult a physician — GERD can damage the esophagus over time if "
            "untreated."
        ),
    },
    {
        "user": "What is lactose intolerance and how does it affect my nutrition?",
        "assistant": (
            "Lactose intolerance results from insufficient production of the enzyme lactase, which "
            "breaks down lactose (milk sugar) in the small intestine. Undigested lactose passes to "
            "the colon where bacteria ferment it, causing bloating, gas, and diarrhea. Most lactose-"
            "intolerant people can tolerate small amounts of dairy, and fermented products like yogurt "
            "and hard cheeses are often better tolerated because some lactose is pre-digested. If you "
            "avoid dairy entirely, ensure adequate calcium from other sources — fortified plant milks, "
            "leafy greens, tofu — to support bone health and muscle function."
        ),
    },
    {
        "user": "What is celiac disease and how does it affect athletic performance?",
        "assistant": (
            "Celiac disease is an autoimmune condition where gluten triggers immune attacks on the "
            "small intestine's villi, impairing nutrient absorption. Athletes with undiagnosed celiac "
            "often experience unexplained fatigue, poor recovery, and iron-deficiency anemia — all "
            "directly impairing performance. After diagnosis, a strict lifelong gluten-free diet allows "
            "the intestine to heal within months, restoring absorption. Gluten-free grains like rice, "
            "quinoa, oats (certified GF), and corn are good carbohydrate sources for athletic "
            "performance. Always work with a registered dietitian to ensure nutritional completeness."
        ),
    },

    # ─── CHAPTER 5: Carbohydrates ─────────────────────────────────────────────

    {
        "user": "Why are carbohydrates called protein sparing?",
        "assistant": (
            "When carbohydrate intake is adequate, the body preferentially uses glucose for energy and "
            "spares amino acids for their primary role: building and repairing muscle tissue, enzymes, "
            "hormones, and immune proteins. If carbohydrate intake is too low, the liver produces glucose "
            "from amino acids through gluconeogenesis — effectively breaking down muscle to fuel the brain. "
            "For athletes trying to build or preserve muscle, adequate carbohydrate intake is just as "
            "important as protein intake. Low-carb diets that sacrifice performance are counterproductive "
            "for anyone with hypertrophy or strength goals."
        ),
    },
    {
        "user": "What is the minimum amount of carbohydrate the brain needs per day?",
        "assistant": (
            "The RDA for carbohydrates is 130 grams per day — this represents the minimum needed to "
            "fuel the brain and central nervous system, which rely almost exclusively on glucose. "
            "Active individuals and athletes typically need far more: 3–12 grams per kg of body weight "
            "depending on training intensity and duration. Consistently consuming below 130 g/day forces "
            "the liver into gluconeogenesis, breaking down muscle protein for glucose, and may impair "
            "cognitive function, mood, and exercise capacity."
        ),
    },
    {
        "user": "What are soluble and insoluble fiber and what are their benefits?",
        "assistant": (
            "Soluble fiber forms a gel-like substance when mixed with water in the GI tract — it slows "
            "digestion, reduces blood glucose spikes after meals, and binds cholesterol in the intestine "
            "preventing its reabsorption, thereby lowering LDL cholesterol. Good sources include oats, "
            "beans, and apples. Insoluble fiber does not dissolve — it adds bulk to stool, speeds transit "
            "time through the colon, and helps prevent constipation and diverticular disease. Both types "
            "support gut microbiome health and are associated with reduced risk of colon cancer, heart "
            "disease, and type 2 diabetes."
        ),
    },
    {
        "user": "How does insulin regulate blood glucose after I eat a high-carb meal?",
        "assistant": (
            "After a high-carb meal, blood glucose rises and the pancreas releases insulin. Insulin "
            "signals GLUT4 transporters in muscle and fat cells to move to the cell membrane and pull "
            "glucose from the bloodstream into cells for energy or storage as glycogen. The liver also "
            "stores glucose as glycogen when insulin is elevated. Once blood glucose returns to normal "
            "(70–100 mg/dL), insulin secretion slows. This negative feedback loop tightly controls "
            "blood glucose. In type 2 diabetes, cells become resistant to insulin's signal, causing "
            "glucose to accumulate in the blood despite adequate insulin production."
        ),
    },
    {
        "user": "What should I eat before a workout?",
        "assistant": (
            "With 2–4 hours before training, eat a balanced meal relatively high in carbohydrates with "
            "some protein and moderate fat — a rice bowl with lean protein, or a turkey sandwich with "
            "fruit. Within 1–2 hours, shift to a smaller, higher-carb snack with moderate protein and "
            "minimal fat and fiber: Greek yogurt and banana, or eggs and toast. In the final hour, "
            "choose quick-digesting carbs that are easy on the stomach: a sports drink, applesauce, "
            "or crackers. Avoid unfamiliar foods before training — practice your pre-workout nutrition "
            "routine during training sessions first."
        ),
    },
    {
        "user": "How many grams of carbohydrate should I consume during a workout over 1 hour?",
        "assistant": (
            "For workouts lasting 1–2.5 hours at moderate to high intensity, aim for 30–60 grams of "
            "carbohydrate per hour. For endurance events exceeding 2.5 hours, you can increase to up "
            "to 90 grams per hour using a glucose-fructose mixture — the combination allows oxidation "
            "rates beyond the 60 g/hr ceiling of glucose alone. Keep carbohydrate sources simple and "
            "fast-digesting during exercise: gels, sports drinks, chews, or diluted fruit juice. Start "
            "fueling early in the session before fatigue sets in — once fatigue begins, GI absorption "
            "slows and you cannot catch up."
        ),
    },
    {
        "user": "What should I eat after a workout to maximize recovery?",
        "assistant": (
            "Consume 0.75–1.5 grams of carbohydrate per kg of body weight as soon as possible after "
            "training — within 30–60 minutes is ideal. Pair it with protein: the combination of "
            "carbohydrate and protein together enhances glycogen resynthesis and muscle protein synthesis "
            "better than carbs alone. Delaying post-workout carbohydrate by several hours slows glycogen "
            "resynthesis by up to 50%. Good recovery food examples: chocolate milk, Greek yogurt with "
            "fruit, rice with chicken, or a turkey sandwich. Liquid options like a protein shake with "
            "banana are convenient when appetite is low right after training."
        ),
    },
    {
        "user": "How do type 1 and type 2 diabetes differ?",
        "assistant": (
            "Type 1 diabetes is an autoimmune disease where the immune system destroys the insulin-"
            "producing beta cells of the pancreas — the person produces no insulin and requires "
            "external insulin injections or a pump to survive. Type 2 diabetes, which accounts for "
            "90–95% of cases, occurs when cells become resistant to insulin or the pancreas cannot "
            "produce enough — blood glucose builds up gradually over years before diagnosis. Type 2 "
            "is strongly linked to excess body fat, inactivity, and poor diet, and can often be "
            "prevented or reversed through lifestyle changes. Both types require careful attention to "
            "carbohydrate intake and exercise timing around meals."
        ),
    },
    {
        "user": "What are the carbohydrate recommendations for different activity levels?",
        "assistant": (
            "Carbohydrate needs scale directly with training intensity and duration. For low-intensity "
            "or skill-based activities (golf, light yoga): 3–5 g/kg/day. For moderate to high intensity "
            "non-endurance training (strength, team sports): 5–7 g/kg/day. For high-intensity endurance "
            "lasting 1–3 hours daily: 6–10 g/kg/day. For extreme endurance (4–5 hours/day at moderate "
            "to high intensity): 8–12 g/kg/day. A 180 lb (82 kg) intermediate lifting 4 days/week would "
            "need approximately 410–575 grams of carbohydrate per day — far above the typical dietary "
            "guidance of 130 g minimum."
        ),
    },

    # ─── CHAPTER 6: Lipids ────────────────────────────────────────────────────

    {
        "user": "What is the difference between saturated and unsaturated fat?",
        "assistant": (
            "Saturated fatty acids have no carbon-carbon double bonds — their chains are fully 'saturated' "
            "with hydrogen atoms, making them pack tightly together and solid at room temperature (butter, "
            "coconut oil, red meat fat). Unsaturated fatty acids have one or more double bonds, creating "
            "a kinked chain that prevents tight packing, making them liquid at room temperature (olive oil, "
            "avocado, fish). Saturated fats raise LDL cholesterol and increase cardiovascular disease risk "
            "when consumed in excess. Unsaturated fats — particularly monounsaturated and polyunsaturated — "
            "lower LDL and reduce inflammation, supporting heart health."
        ),
    },
    {
        "user": "What are omega-3 and omega-6 fatty acids and why does their balance matter?",
        "assistant": (
            "Omega-3 and omega-6 are essential polyunsaturated fatty acids the body cannot make — both "
            "must come from diet. Omega-6 (found in most vegetable oils, nuts, and processed foods) "
            "produces eicosanoids that increase inflammation, blood pressure, and immune response. "
            "Omega-3 (found in fatty fish, flaxseed, walnuts) produces eicosanoids that reduce those "
            "same responses. Modern Western diets are heavily skewed toward omega-6, which chronically "
            "elevates inflammation and is linked to heart disease, arthritis, and cancer. Aim to eat "
            "2 servings of fatty fish per week and use flaxseed or chia as plant-based omega-3 sources "
            "to improve balance."
        ),
    },
    {
        "user": "What are trans fats and why were they removed from food?",
        "assistant": (
            "Trans fats are created by partially hydrogenating liquid vegetable oils — a process that "
            "adds hydrogen atoms, making them solid and shelf-stable for use in processed foods like "
            "cookies, crackers, and fried foods. Despite being unsaturated, trans fats behave like "
            "saturated fats in the body but are more potent — they simultaneously raise LDL ('bad') "
            "cholesterol and lower HDL ('good') cholesterol, dramatically increasing cardiovascular "
            "disease risk. The FDA banned partially hydrogenated oils from the U.S. food supply starting "
            "January 2020. Check ingredients for 'hydrogenated' or 'partially hydrogenated' oils, which "
            "may still appear in older packaged products."
        ),
    },
    {
        "user": "What is the difference between LDL and HDL cholesterol?",
        "assistant": (
            "LDL (low-density lipoprotein) carries cholesterol from the liver to body tissues — when "
            "elevated, it deposits cholesterol in artery walls, forming plaques that cause atherosclerosis "
            "and heart attacks. An LDL below 100 mg/dL is ideal. HDL (high-density lipoprotein) performs "
            "reverse cholesterol transport — it picks up cholesterol from tissues and artery walls and "
            "returns it to the liver for disposal. Higher HDL is protective; below 40 mg/dL for men "
            "or 50 mg/dL for women signals elevated heart disease risk. Regular exercise, omega-3 intake, "
            "and replacing saturated fat with unsaturated fat all raise HDL and lower LDL."
        ),
    },
    {
        "user": "What is atherosclerosis and how does it develop?",
        "assistant": (
            "Atherosclerosis is the progressive hardening and narrowing of arteries caused by plaque "
            "buildup. It begins with injury to the inner arterial lining from factors like high blood "
            "pressure, high cholesterol, smoking, or chronic inflammation. LDL particles enter the "
            "damaged wall, become oxidized, and trigger an immune response — macrophages engulf the "
            "oxidized LDL forming foam cells. This builds into plaques of cholesterol, calcium, and "
            "fibrin that narrow the artery, restrict blood flow, and can rupture — triggering clots "
            "that cause heart attacks or strokes. The disease begins in childhood and progresses "
            "silently for decades, which is why early lifestyle habits matter."
        ),
    },
    {
        "user": "What is the DASH diet and who should follow it?",
        "assistant": (
            "DASH (Dietary Approaches to Stop Hypertension) is an evidence-based eating pattern that "
            "reduces blood pressure by limiting sodium to 2300 mg/day (or 1500 mg for high-risk "
            "individuals) while emphasizing whole grains, fruits, vegetables, low-fat dairy, lean "
            "meats, fish, nuts, and legumes. It limits saturated fat under 7% of calories and restricts "
            "sweets, alcohol, and sodium-heavy processed foods. DASH also reduces risk of stroke, "
            "diabetes, osteoporosis, and certain cancers. Anyone with hypertension, metabolic syndrome, "
            "or high cardiovascular disease risk should consider adopting it — but it is a healthy "
            "pattern for most people regardless of blood pressure."
        ),
    },
    {
        "user": "What is metabolic syndrome?",
        "assistant": (
            "Metabolic syndrome is diagnosed when a person has three or more of the following: "
            "waist circumference above 102 cm (men) or 88 cm (women); triglycerides ≥150 mg/dL; "
            "blood pressure ≥130/85 mm Hg; HDL cholesterol below 40 mg/dL (men) or 50 mg/dL (women); "
            "or fasting blood sugar ≥100 mg/dL. Having metabolic syndrome doubles the risk of heart "
            "disease and increases diabetes risk fivefold compared to those without it. Fortunately, "
            "all five components are modifiable through dietary change, regular exercise, and weight "
            "management — it does not require medication in most cases."
        ),
    },

    # ─── CHAPTER 7: Proteins ──────────────────────────────────────────────────

    {
        "user": "How much protein do I need per day if I lift weights?",
        "assistant": (
            "For strength-trained athletes, the evidence-based range is 1.6–2.0 grams of protein per "
            "kg of bodyweight per day. Protein intakes beyond 2.0 g/kg do not produce additional "
            "muscle protein synthesis benefit for most people, though intakes up to 3.0 g/kg appear "
            "safe for healthy adults. A 180 lb (82 kg) intermediate lifter targeting hypertrophy "
            "should consume approximately 131–164 grams of protein daily. Distribute it across 3–5 "
            "meals to maximize muscle protein synthesis, aiming for at least 2.5–3 grams of leucine "
            "per meal — achievable with 30–40 grams of quality protein per sitting."
        ),
    },
    {
        "user": "What are essential amino acids and which foods contain all of them?",
        "assistant": (
            "Essential amino acids are the nine amino acids the body cannot synthesize and must obtain "
            "from food: histidine, isoleucine, leucine, lysine, methionine, phenylalanine, threonine, "
            "tryptophan, and valine. Foods containing all nine are called complete proteins — these "
            "include all animal products (meat, poultry, fish, eggs, dairy) as well as a few plant "
            "foods: soybeans and quinoa. Most plant proteins are incomplete — they lack or are low in "
            "one or more essential amino acids. Vegetarians and vegans can meet all amino acid needs "
            "through protein complementation, combining foods like rice and beans across the day."
        ),
    },
    {
        "user": "Can I build muscle on a plant-based diet?",
        "assistant": (
            "Yes, absolutely — but it requires more planning. Plant proteins have lower digestibility "
            "than animal proteins, so vegetarians should increase total protein intake by approximately "
            "10% above standard recommendations. Lysine is the limiting amino acid in most plant diets — "
            "prioritize legumes (beans, lentils, tofu, tempeh) and quinoa, which are good lysine sources. "
            "Combine grains and legumes throughout the day for full amino acid coverage. Creatine "
            "supplementation is particularly beneficial for vegetarians, as they tend to have lower "
            "baseline muscle creatine stores than omnivores and respond more strongly to supplementation."
        ),
    },
    {
        "user": "What is protein complementation and do I need to combine foods at every meal?",
        "assistant": (
            "Protein complementation means pairing plant foods that are deficient in different amino "
            "acids so together they provide a complete amino acid profile — for example, grains (low in "
            "lysine) paired with legumes (low in methionine). Rice and beans, hummus and pita, and tofu "
            "with cashews are classic combinations. The good news: you do not need to eat complementary "
            "proteins at the same meal — consuming them within the same day is sufficient for your amino "
            "acid pool to be adequately stocked. Focus on variety across the whole day rather than "
            "perfectly balancing each plate."
        ),
    },
    {
        "user": "Can eating too much protein damage my kidneys?",
        "assistant": (
            "For healthy adults with normal kidney function, current research shows that intakes above "
            "2.0 g/kg — even as high as 3.0+ g/kg — do not damage the kidneys or liver. However, if "
            "you have pre-existing kidney disease, high protein intake can accelerate kidney decline and "
            "should be discussed with your physician. The main practical concern with very high protein "
            "diets is the risk of insufficient carbohydrate intake, reduced dietary variety, and "
            "dehydration — protein metabolism produces urea that must be excreted in urine, increasing "
            "fluid needs. Drink adequate water and maintain a balanced diet rather than maximizing "
            "protein at the expense of other nutrients."
        ),
    },
    {
        "user": "Are protein supplements better than whole food protein sources?",
        "assistant": (
            "No — protein and amino acid supplements are not more effective than food for gaining lean "
            "mass when total energy and protein are adequate. The Academy of Nutrition and Dietetics, "
            "ACSM, and Dietitians of Canada all concur on this. Whole food proteins come packaged with "
            "vitamins, minerals, fiber, and other beneficial compounds that supplements lack. Protein "
            "powders and bars are useful tools for convenience — ideal as a post-workout snack when "
            "whole food isn't available — but they should supplement your diet, not replace meals. "
            "If cost matters, a peanut butter sandwich on whole-grain bread with milk costs less and "
            "provides comparable protein and more micronutrients than most commercial protein bars."
        ),
    },
    {
        "user": "How much protein does an older adult need to prevent muscle loss?",
        "assistant": (
            "Older adults benefit from slightly higher protein intake than the standard 0.8 g/kg RDA "
            "to counteract age-related muscle loss (sarcopenia), which begins as early as the 40s. "
            "Research supports 1.1–1.2 g/kg daily as a minimum to help preserve muscle mass in older "
            "adults who are not actively resistance training. Those who do resistance train regularly "
            "may benefit from 1.6–2.0 g/kg — the same range as younger athletes. Strength training "
            "paired with adequate protein is the most effective combination for slowing sarcopenia. "
            "Increasing protein at every meal, particularly breakfast, helps meet daily targets."
        ),
    },
    {
        "user": "What happens to protein in the body when I do not eat enough carbohydrates?",
        "assistant": (
            "When carbohydrate intake is insufficient, the liver performs gluconeogenesis — converting "
            "amino acids from muscle tissue into glucose to maintain blood sugar for brain function. "
            "This is why low-carb diets that are also low in calories make it very difficult to "
            "maintain or build muscle mass, regardless of how much protein you eat. The protein you "
            "consume gets redirected toward energy production rather than muscle repair and growth. "
            "For anyone with hypertrophy or strength goals, adequate carbohydrate intake is essential "
            "to ensure dietary protein actually goes toward building muscle."
        ),
    },

    # ─── CHAPTER 8: Metabolism ────────────────────────────────────────────────

    {
        "user": "What is ATP and why does every exercise discussion mention it?",
        "assistant": (
            "ATP (adenosine triphosphate) is the universal energy currency of the body — every muscular "
            "contraction, nerve impulse, and cellular process runs on ATP. When the high-energy bond "
            "between the second and third phosphate groups is broken, energy is released to do work. "
            "The body stores very little ATP — only enough for a few seconds of maximal effort — so "
            "it must be continuously resynthesized from carbohydrates, fats, creatine phosphate, and "
            "to a lesser extent protein. Understanding the three energy systems explains why different "
            "exercise intensities and durations feel different and require different nutrition strategies."
        ),
    },
    {
        "user": "What are the three energy systems and when does each one dominate?",
        "assistant": (
            "The ATP-CP (phosphagen) system dominates for 0–10 seconds of maximal effort — sprints, "
            "heavy lifts, explosive jumps. It uses creatine phosphate to regenerate ATP nearly "
            "instantaneously without oxygen, but stores deplete in under 10 seconds. Glycolysis takes "
            "over from roughly 10 seconds to 2 minutes at high intensity — it breaks down glucose "
            "anaerobically, producing ATP quickly but generating lactate. Aerobic metabolism (Krebs "
            "cycle + electron transport chain) dominates from 2 minutes onward, producing 30–33 ATP "
            "per glucose molecule using oxygen — it is slower to activate but supports hours of "
            "sustained activity. All three systems operate simultaneously, with one predominating."
        ),
    },
    {
        "user": "How does creatine supplementation work and who benefits most?",
        "assistant": (
            "Creatine supplementation increases muscle creatine phosphate stores, allowing more "
            "rapid ATP regeneration during short, explosive efforts. An average diet leaves muscles "
            "60–80% saturated with creatine; supplementing with 3–5 g/day over 4 weeks maximizes "
            "stores. This supports greater training volume in power activities — more reps, heavier "
            "loads — which drives greater strength and muscle gains over time. Vegetarians benefit "
            "most because they have lower baseline creatine stores. Women also tend to have lower "
            "baseline stores than men and respond well. Safety is well-established: 3–5 g/day of "
            "creatine monohydrate is safe for healthy adults, including long-term use."
        ),
    },
    {
        "user": "Why does fat burning slow down at the start of a workout?",
        "assistant": (
            "Fat oxidation requires a multi-step process: triglycerides must be broken down via "
            "lipolysis, fatty acids transported to muscle cells, then processed through beta-oxidation "
            "before entering aerobic metabolism. This takes time — there is typically a 10–20 minute "
            "'lag' at the onset of exercise before fat metabolism fully ramps up. At rest, fat provides "
            "about 85% of energy. As intensity increases, the body shifts toward carbohydrates because "
            "glycolysis is faster. At true high intensity, carbohydrates provide nearly all fuel. "
            "Better aerobic fitness shortens this lag time and increases the proportion of fat used at "
            "any given submaximal intensity — fit people genuinely are better fat burners."
        ),
    },
    {
        "user": "What is lactate and is it responsible for muscle soreness?",
        "assistant": (
            "Lactate is produced during high-intensity glycolysis when pyruvate is converted to "
            "maintain NAD+ availability — it is not a metabolic waste product but an important "
            "intermediary that can be reconverted to pyruvate and used for aerobic energy production "
            "in the heart, liver, and slow-twitch muscle fibers. Lactate is rapidly cleared during "
            "and after exercise. Importantly, lactate is NOT the cause of delayed-onset muscle soreness "
            "(DOMS) — DOMS results from microscopic muscle damage and subsequent inflammation occurring "
            "24–72 hours after unfamiliar or eccentric exercise, by which point lactate has long been "
            "cleared from the body."
        ),
    },
    {
        "user": "How does my body use fat versus carbohydrate during a long run?",
        "assistant": (
            "At low to moderate intensity (roughly 50–65% of max effort), fat and carbohydrate each "
            "contribute about equally to energy production. As pace increases above moderate intensity, "
            "the contribution shifts progressively toward carbohydrate. At a well-conditioned endurance "
            "athlete's race pace, carbohydrate provides the majority of fuel but glycogen stores are "
            "limited — typically 24 hours at moderate intensity, significantly less at race pace. "
            "Well-trained endurance athletes metabolize fat more efficiently and preserve glycogen "
            "longer than untrained individuals. This is why long slow distance training and carbohydrate "
            "loading before races both matter for endurance performance."
        ),
    },
    {
        "user": "What is the ATP-CP system and how does rest between sets affect it?",
        "assistant": (
            "The ATP-CP system uses stored creatine phosphate to regenerate ATP in a single enzymatic "
            "step — it is the fastest energy system but lasts only 5–10 seconds at maximal effort. "
            "After a maximal set, creatine phosphate stores can be partially replenished in 1–2 minutes "
            "of rest, which is why 2–3 minute rest periods between heavy compound sets improve "
            "subsequent performance. Rephosphorylation of CP requires oxygen, meaning aerobic fitness "
            "improves CP recovery rate. Insufficient rest between sets means each subsequent set is "
            "increasingly powered by glycolysis — effective for hypertrophy but suboptimal for "
            "maximal strength or power output."
        ),
    },

    # ─── CHAPTER 9: Weight & Body Image ──────────────────────────────────────

    {
        "user": "What are the limitations of using BMI to assess my health?",
        "assistant": (
            "BMI is a simple weight-to-height ratio that correlates reasonably with body fat at the "
            "population level, but it does not measure body composition. Athletes routinely fall into "
            "'overweight' or 'obese' BMI categories because muscle is denser than fat — their health "
            "risk is not elevated. Conversely, 'skinny-fat' individuals with normal BMI but high body "
            "fat percentage and low muscle mass face elevated chronic disease risk. BMI also has lower "
            "accuracy for certain ethnic groups whose health risk thresholds differ from the populations "
            "used to establish the ranges. Use BMI as a rough population-level screening tool, not as "
            "a definitive assessment of individual health."
        ),
    },
    {
        "user": "What does waist circumference tell me that BMI doesn't?",
        "assistant": (
            "Waist circumference measures abdominal obesity — the accumulation of visceral fat around "
            "internal organs — which carries greater metabolic risk than fat stored around the hips and "
            "thighs. A waist above 40 inches for men or 35 inches for women defines abdominal obesity "
            "and is a stronger predictor of metabolic syndrome, type 2 diabetes, and cardiovascular "
            "disease than BMI alone. Two people can have identical BMIs but very different waist "
            "circumferences and disease risks. Measuring waist circumference alongside BMI gives a "
            "more complete picture of metabolic health — and tracking changes in waist over time "
            "reflects visceral fat loss better than scale weight."
        ),
    },
    {
        "user": "What is the most accurate way to measure body fat percentage?",
        "assistant": (
            "DEXA (dual X-ray absorptiometry) is currently the gold standard — it measures bone "
            "density, fat mass, and lean mass regionally with 2–3% margin of error while controlling "
            "for bone mineral density. Hydrostatic (underwater) weighing and air displacement "
            "plethysmography (Bod Pod) are also highly accurate. Bioelectrical impedance (BIA) is "
            "convenient and inexpensive but sensitive to hydration status — results vary significantly "
            "based on when you last ate, drank, or exercised. Skinfold calipers require a trained "
            "technician but are portable and useful for tracking changes over time when done "
            "consistently. For tracking progress, consistency in method matters more than method accuracy."
        ),
    },
    {
        "user": "Can someone be overweight and still be healthy?",
        "assistant": (
            "Research from the Cooper Institute shows that overweight men and women who are "
            "cardiovascularly fit have mortality rates not significantly different from lean fit "
            "individuals — and fit overweight people have roughly half the mortality risk of unfit "
            "lean people. Cardiovascular fitness is a better predictor of mortality than BMI. This does "
            "not mean obesity has no risks — it is a meaningful risk factor for many chronic diseases. "
            "But it means that regular exercise, independent of weight loss, provides substantial health "
            "benefits. Rather than obsessing over the scale, monitoring health markers — blood pressure, "
            "cholesterol, blood glucose, aerobic fitness — is a more actionable and motivating approach."
        ),
    },
    {
        "user": "Why do very low calorie diets fail long-term for weight loss?",
        "assistant": (
            "Very low calorie diets trigger starvation physiology: the body slows metabolic rate by up "
            "to 20% through adaptive thermogenesis, breaks down lean muscle mass for fuel (further "
            "reducing RMR), depletes glycogen and water rapidly creating the illusion of fast fat loss, "
            "and increases hunger hormones. Most of the initial weight loss on a crash diet is water "
            "and glycogen, not fat. When normal eating resumes, glycogen and water are rapidly restored "
            "and the now-lower metabolic rate means more calories are stored as fat than before. "
            "The evidence-based approach is a modest 300–500 kcal deficit with adequate protein and "
            "resistance training to preserve muscle while slowly reducing fat."
        ),
    },
    {
        "user": "How do genetics affect my ability to lose weight or build muscle?",
        "assistant": (
            "Genetics influences muscle fiber type distribution, hormonal environment, fat storage "
            "patterns, and baseline metabolic rate — all of which affect how quickly you respond to "
            "training and diet. Having a parent who is obese doubles your risk; if both parents are "
            "obese it rises fivefold. However, genetic predisposition is not genetic destiny — 'genetics "
            "loads the gun, lifestyle pulls the trigger.' People genetically prone to obesity who maintain "
            "an active lifestyle and nutritious diet significantly reduce their disease risk. Realistic "
            "expectations, consistent effort, and comparison to your own previous performance rather "
            "than others' results are key to long-term success regardless of genetics."
        ),
    },

    # ─── PRACTICAL APPLIED QUESTIONS ─────────────────────────────────────────

    {
        "user": "I want to lose fat and build muscle at the same time. Is that possible?",
        "assistant": (
            "Body recomposition — simultaneous fat loss and muscle gain — is most achievable in "
            "beginners, people returning after a layoff, and those with significant fat to lose. "
            "The strategy requires a slight caloric deficit (200–300 kcal/day), high protein intake "
            "(1.6–2.2 g/kg), consistent progressive resistance training, and adequate sleep. The "
            "rate of change is slower than pursuing each goal separately, but the result is sustained. "
            "Advanced lean athletes face a harder trade-off and typically get better results alternating "
            "structured gaining and cutting phases rather than trying to do both simultaneously."
        ),
    },
    {
        "user": "What supplements actually have scientific evidence behind them for muscle building?",
        "assistant": (
            "Creatine monohydrate has the most consistent and robust evidence base — it reliably "
            "improves strength, power, and lean mass across hundreds of studies and is safe for "
            "long-term use at 3–5 g/day. Protein supplementation helps when total daily protein from "
            "food falls short of your target (1.6–2.0 g/kg). Caffeine improves performance and "
            "training volume. Beyond these three, evidence for most other supplements is weak, mixed, "
            "or industry-funded. Many sports supplements contain undeclared substances — choose only "
            "third-party certified products (NSF for Sport or Informed Sport) if you are subject to "
            "drug testing."
        ),
    },
    {
        "user": "Should I train fasted in the morning for better fat loss?",
        "assistant": (
            "Fasted training does burn a slightly higher proportion of fat during the session itself, "
            "but total fat loss over 24 hours is not significantly different from fed training when "
            "total calories are matched. What fasted training does cost you is performance — lower "
            "glycogen availability means lower training intensity, which reduces total calorie burn "
            "and may impair muscle retention over time. For fat loss with preserved muscle mass, "
            "a small pre-workout protein and carbohydrate snack outperforms fasted sessions. If you "
            "genuinely prefer fasted training and performance is not impacted, it is fine — the "
            "overall diet context matters far more than training timing."
        ),
    },
    {
        "user": "How does sleep affect muscle building and fat loss?",
        "assistant": (
            "Sleep is when the majority of growth hormone release and muscle protein synthesis occur — "
            "chronic sleep deprivation blunts anabolic hormone levels, increases cortisol (catabolic), "
            "reduces testosterone, and impairs insulin sensitivity. Studies show that dieting on "
            "insufficient sleep results in a greater proportion of lean mass lost versus fat mass "
            "compared to adequate sleep. Sleep deprivation also increases appetite hormones (ghrelin) "
            "and reduces satiety hormones (leptin), making caloric control harder. Seven to nine hours "
            "of quality sleep per night is not optional for physique goals — it is a core part of "
            "the recovery program alongside nutrition and training."
        ),
    },
    {
        "user": "How much fiber should I eat per day?",
        "assistant": (
            "The general recommendation is 25 grams per day for women and 38 grams for men, though "
            "most Americans consume only 15 grams daily. Athletes with high calorie intakes naturally "
            "consume more fiber if eating whole foods. Fiber from whole food sources — vegetables, "
            "legumes, whole grains, fruits, nuts, and seeds — provides the greatest benefit. Increase "
            "fiber intake gradually and drink plenty of water alongside it to avoid bloating and "
            "discomfort. High fiber intake improves gut health, controls blood glucose, reduces LDL "
            "cholesterol, increases satiety, and lowers risk of colon cancer and heart disease."
        ),
    },
    {
        "user": "What are the signs that I need to take a deload week?",
        "assistant": (
            "A deload is warranted when you notice multiple signs of accumulated fatigue: strength "
            "or performance declining across consecutive sessions, persistent joint pain or unusual "
            "soreness, poor sleep quality, low motivation or dreading training, and elevated resting "
            "heart rate. As a scheduled approach, programming a deload every 4–8 weeks (reduce volume "
            "by 40–60%, keep weight the same) prevents fatigue from masking fitness gains. The "
            "difference between a productive deload and wasted time is keeping intensity (load) "
            "similar — the goal is systemic recovery, not a vacation from progressive overload."
        ),
    },
    {
        "user": "I am a vegetarian. Should I take creatine?",
        "assistant": (
            "Yes — creatine supplementation is particularly beneficial for vegetarians. Because creatine "
            "is found almost exclusively in meat and fish, vegetarians have significantly lower baseline "
            "muscle creatine stores than omnivores. Research shows that vegetarians experience greater "
            "increases in muscle creatine levels from supplementation and stronger performance and "
            "strength responses. Start with 3–5 grams of creatine monohydrate per day — no loading "
            "phase required if you are patient (full saturation in 4 weeks). Take it consistently "
            "at any time of day; pairing it with a protein-and-carbohydrate meal may slightly enhance "
            "uptake due to the insulin response."
        ),
    },
    {
        "user": "What is the glycemic index and does it matter for my training?",
        "assistant": (
            "The glycemic index (GI) ranks carbohydrate foods by how rapidly they raise blood glucose "
            "— high GI foods (white rice, sports drinks, white bread) digest quickly; low GI foods "
            "(oats, legumes, sweet potato) release glucose slowly. For pre-workout meals 2+ hours out, "
            "lower GI foods promote sustained energy without the crash. Around workouts — immediately "
            "before, during, and immediately after — high GI foods are advantageous: faster digestion "
            "means faster glucose delivery to muscles when the window is short. For overall health "
            "and blood sugar management, lower GI foods throughout the rest of the day support better "
            "insulin sensitivity and satiety."
        ),
    },
    {
        "user": "Is intermittent fasting effective for fat loss?",
        "assistant": (
            "Intermittent fasting (IF) is an effective fat loss strategy primarily because it "
            "simplifies caloric restriction — fewer eating windows naturally reduces total calorie "
            "intake for many people. Controlled studies show that when calories and protein are matched, "
            "IF produces similar fat loss to continuous caloric restriction. Where it can underperform "
            "is in muscle retention: compressed eating windows make it harder to hit protein targets "
            "and optimally distribute protein across multiple meals for muscle protein synthesis. "
            "If you enjoy IF and can consistently hit your protein and calorie targets within your "
            "eating window, it is a valid approach — the best diet is the one you can adhere to."
        ),
    },
    {
        "user": "Why do I need more carbs on training days versus rest days?",
        "assistant": (
            "On training days, your muscles use glycogen as primary fuel — especially during moderate "
            "to high intensity work — and post-exercise, carbohydrate is needed to replenish those "
            "depleted stores and trigger the insulin response that drives nutrient uptake into muscle. "
            "On rest days, energy expenditure is lower and glycogen is not being depleted, so total "
            "carbohydrate needs are reduced. A practical approach is carbohydrate periodization: "
            "higher carb on training days (5–7+ g/kg) and lower carb on rest days (3–4 g/kg), keeping "
            "protein constant. This approach optimizes performance and body composition without requiring "
            "a strict cutting diet every day."
        ),
    },
    {
        "user": "How do I know if I am eating enough protein at each meal?",
        "assistant": (
            "Each meal should provide at least 2.5–3 grams of leucine — the key anabolic amino acid — "
            "to maximally stimulate muscle protein synthesis. This is typically achieved with 30–40 grams "
            "of complete animal protein per meal, or slightly more from plant sources due to lower "
            "digestibility. Practically: a palm-sized portion of chicken, fish, beef, or 1.5–2 cups of "
            "Greek yogurt or cottage cheese per meal generally meets this threshold. If you are eating "
            "plant proteins, combine sources (legumes + grain or legumes + seeds) and aim for the "
            "higher end of the 30–40 gram range since plant proteins are less efficiently absorbed."
        ),
    },
    {
        "user": "What is the difference between overtraining syndrome and normal training fatigue?",
        "assistant": (
            "Normal training fatigue is short-term — you feel tired after sessions but recover well "
            "within 24–72 hours and performance improves over weeks. Overtraining syndrome is a "
            "pathological state where accumulated fatigue exceeds recovery capacity, leading to "
            "performance decrements lasting weeks to months despite reduced training. Signs include "
            "persistent performance decline across all lifts, elevated resting heart rate, disrupted "
            "sleep, mood changes, frequent illness, and loss of training motivation. Management "
            "requires extended rest, aggressive caloric and protein intake, and stress reduction. "
            "Preventing overtraining is far easier than recovering from it — program deloads and "
            "respect recovery windows."
        ),
    },
    {
        "user": "How important is hydration for exercise performance?",
        "assistant": (
            "Even mild dehydration (1–2% of body weight) measurably impairs both aerobic and cognitive "
            "performance. A 2% dehydration in a 180 lb athlete equals just 3.6 lbs of fluid loss — "
            "easily achieved in a long or hot training session. Monitor urine color throughout the day: "
            "pale straw yellow indicates adequate hydration; dark yellow signals dehydration. Drink "
            "16–24 oz of water 2 hours before training, sip during exercise, and replace lost fluids "
            "after. For sessions over 60–90 minutes in heat, electrolyte replacement (particularly "
            "sodium) becomes important to prevent hyponatremia from drinking plain water excessively."
        ),
    },
    {
        "user": "What is RPE and how do I use it to guide training intensity?",
        "assistant": (
            "RPE (Rating of Perceived Exertion) or the newer RIR (Reps in Reserve) system quantifies "
            "effort relative to your maximum on a given day. RPE 10 or 0 RIR means all-out — you could "
            "not do another rep. RPE 8 or 2 RIR means you had 2 reps left in the tank. For hypertrophy, "
            "most working sets should be at RPE 7–9 (1–3 RIR) — close to failure but not maxing out "
            "every set, which accumulates excessive fatigue. RPE adjusts automatically for daily "
            "fluctuations in readiness — if you slept poorly or are stressed, reaching RPE 8 may "
            "require less weight than usual, and that is appropriate. Train the effort, not the number."
        ),
    },
    {
        "user": "Should I be concerned about added sugar if I am an active athlete?",
        "assistant": (
            "For athletes with high calorie expenditure, some added sugar is often practical and "
            "appropriate — especially from sports drinks, gels, and recovery foods that provide "
            "fast glucose around training. The Dietary Guidelines' recommendation to keep added sugar "
            "under 10% of calories applies primarily to sedentary and moderately active individuals "
            "where empty calories displace nutrients. For a 3000-calorie athlete, 10% equals 300 "
            "calories of added sugar — significantly more flexibility. The concern with excess added "
            "sugar is displacement of nutrient-dense foods and contribution to obesity in those not "
            "burning those calories through training. Contextualize total diet quality, not single nutrients."
        ),
    },
    {
        "user": "What is the best source of omega-3 if I do not eat fish?",
        "assistant": (
            "If you do not eat fish, plant-based omega-3 sources provide alpha-linolenic acid (ALA): "
            "flaxseeds and flaxseed oil, chia seeds, walnuts, hemp seeds, and canola oil. The challenge "
            "is that conversion of ALA to the active forms DHA and EPA is very inefficient — less than "
            "10% converts. For the full anti-inflammatory benefit without fish, algae-based omega-3 "
            "supplements (algal DHA/EPA) are the most direct option and are what fish themselves eat. "
            "These are well-tolerated and appropriate for vegetarians and vegans. Two to three grams of "
            "combined DHA and EPA per day from algal supplements provides benefits comparable to "
            "regular fish consumption."
        ),
    },
    {
        "user": "How do stimulants like caffeine affect performance and metabolism?",
        "assistant": (
            "Caffeine improves aerobic endurance, strength, and power output through adenosine receptor "
            "blockade — it reduces perception of effort and fatigue. It also temporarily increases "
            "metabolic rate and fat oxidation, though the effect on fat loss from this is modest. "
            "Effective dosing is 3–6 mg per kg of body weight (210–420 mg for a 70 kg person) taken "
            "30–60 minutes before training. Tolerance develops with regular use — cycling off caffeine "
            "periodically (1–2 weeks) restores sensitivity. Avoid caffeine 6+ hours before sleep to "
            "protect sleep quality, which is more valuable to performance than any ergogenic compound."
        ),
    },
    {
        "user": "What foods should I prioritize for heart health?",
        "assistant": (
            "Prioritize fatty fish (2 servings/week for EPA and DHA), whole grains (oats, barley for "
            "soluble fiber that lowers LDL), legumes (beans, lentils for fiber and plant protein), "
            "nuts especially walnuts and almonds (unsaturated fats and fiber), olive oil over butter, "
            "and abundant colorful vegetables and fruits (fiber, potassium, antioxidants). Limit "
            "saturated fat from red and processed meat, trans fats from packaged foods, sodium above "
            "2300 mg/day, added sugars, and alcohol. The DASH and Mediterranean dietary patterns both "
            "have strong evidence for cardiovascular protection and align well with athletic nutrition "
            "needs — they are not restrictive diets but sustainable eating frameworks."
        ),
    },
    {
        "user": "I am a beginner. How do I start losing weight without losing motivation?",
        "assistant": (
            "Start with small, sustainable changes rather than an overhaul: add a 30-minute walk "
            "daily, reduce portions of the highest-calorie foods you eat most often (sugary drinks, "
            "fried foods), and add a vegetable to two meals per day. Target a modest deficit of 300 "
            "kcal/day — small enough to maintain energy for training and daily life, large enough to "
            "produce visible change in 4–6 weeks. Begin strength training 2–3 days per week to "
            "preserve muscle as you lose fat. Track progress weekly by weight, waist circumference, "
            "and how your clothes fit — not daily, as normal fluctuations mislead. Celebrate "
            "process wins (consistent workouts, hitting protein target) not just scale changes."
        ),
    },
    {
        "user": "Does eating late at night cause fat gain?",
        "assistant": (
            "Eating at night does not cause fat gain by itself — total calories over 24 hours determine "
            "body composition, not the timing of when those calories are consumed. The reason late "
            "eating is associated with weight gain is behavioral: nighttime eating often involves "
            "mindless, high-calorie snacking on top of an already adequate daily intake, and tiredness "
            "reduces food decision quality. If your total daily calories are within your target, "
            "a post-dinner protein snack or meal to hit your protein goals is not harmful and may "
            "support muscle protein synthesis overnight. Time your meals around what enables the most "
            "consistent adherence to your daily calorie and protein targets."
        ),
    },
    {
        "user": "What is progressive overload and how do I apply it practically?",
        "assistant": (
            "Progressive overload means systematically increasing the training stimulus over time to "
            "continue driving adaptation — without it, the body has no reason to change. Practically, "
            "the simplest method is adding weight when you complete all prescribed reps with 1–2 reps "
            "in reserve: if you hit 3×10 at RPE 7, add 2.5–5 kg next session. Other forms include "
            "adding a rep or set, reducing rest time, improving range of motion, or slowing the "
            "eccentric. Beginners can add load every session; intermediates progress weekly or "
            "bi-weekly; advanced lifters progress across 4–6 week training blocks. Track every session "
            "in a log — if you cannot see progress on paper, you are not progressively overloading."
        ),
    },
    {
        "user": "How does protein intake affect satiety when trying to lose weight?",
        "assistant": (
            "Protein is the most satiating macronutrient — it suppresses hunger more powerfully than "
            "equivalent calories from carbohydrates or fat through multiple hormonal mechanisms "
            "(increased GLP-1, PYY, and CCK; reduced ghrelin). High protein intake during a caloric "
            "deficit reduces muscle breakdown and keeps metabolic rate higher than low-protein diets. "
            "Practically: starting each meal with a protein source (eggs, chicken, Greek yogurt, "
            "cottage cheese) before eating carbohydrates and fats reduces total caloric intake "
            "without requiring willpower. Aiming for 30–40 grams of protein per meal, 3–5 times per "
            "day, is both optimal for muscle retention and highly effective for controlling hunger."
        ),
    },
    {
        "user": "What is the role of vitamin D in fitness and performance?",
        "assistant": (
            "Vitamin D functions as a hormone rather than a classic vitamin — receptors for it exist "
            "in muscle tissue, and deficiency is associated with impaired muscle function, reduced "
            "strength, increased injury risk, and suppressed immune function. Athletes training indoors "
            "or living at higher latitudes with limited sun exposure are at high risk of deficiency. "
            "Optimal vitamin D levels (50–80 nmol/L serum) support testosterone production, bone "
            "density (critical for absorbing training stress), and immune health. The RDA of 600 IU "
            "is a minimum; many sports medicine practitioners recommend 1000–2000 IU supplementation "
            "for athletes, particularly through winter months. Get a blood test to know your baseline."
        ),
    },
    {
        "user": "What is the best approach to nutrition for a soccer player?",
        "assistant": (
            "Soccer involves all three energy systems — aerobic metabolism for sustained running, "
            "glycolysis for repeated sprints, and ATP-CP for explosive actions like shooting and "
            "jumping — making carbohydrate the primary fuel. Target 5–7 g/kg of carbohydrate daily "
            "on training and match days, with emphasis on glycogen replenishment within 2 hours post-"
            "match via carbohydrate and protein. Pre-match meal (3–4 hours before): moderate "
            "carbohydrate, lean protein, low fat and fiber to minimize GI distress. Hydration with "
            "electrolytes is critical in multi-session days. Protein at 1.4–1.7 g/kg daily supports "
            "the lean mass demands of acceleration and power production."
        ),
    },
    {
        "user": "Why do some people plateau in weight loss even when eating in a deficit?",
        "assistant": (
            "Plateaus happen for several interconnected reasons. As you lose weight, your RMR decreases "
            "because there is less body mass to maintain — so the same caloric intake that created a "
            "deficit initially may no longer do so. Adaptive thermogenesis also reduces non-exercise "
            "activity (unconscious fidgeting and movement) in response to restriction. Food tracking "
            "accuracy also tends to deteriorate over time, with portion creep underestimating intake. "
            "To break a plateau: reassess your actual calorie intake honestly (re-weigh and measure "
            "foods), reduce calories by another 100–200 kcal, or add 20–30 minutes of cardio to "
            "increase expenditure. Patience is also a factor — true plateaus of 3–4 weeks warrant "
            "action; a 1–2 week stall is often normal fluctuation."
        ),
    },

]


def build_jsonl(output_path: str) -> int:
    samples = []
    for qa in QA_PAIRS:
        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": qa["user"]},
                {"role": "assistant", "content": qa["assistant"]},
            ]
        }
        samples.append(record)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return len(samples)


if __name__ == "__main__":
    out = "artifacts/datasets/nutrition_textbook_data.jsonl"
    n = build_jsonl(out)
    print(f"Saved {n} samples -> {out}")
