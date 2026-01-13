import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.awt.geom.GeneralPath;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;
import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.util.List;

public class Main {
    // Main entry point of the program
    public static void main(String[] args) {
        // Path to the target image that will be approximated
        String targetPath = "input.jpg";

        // Evolutionary Algorithm parameters:
        int populationSize = 100;           // Number of individuals in each generation
        int genesPerIndividual = 500;       // Number of geometric shapes per individual
        int generations = 20000;            // Total number of generations to evolve
        double mutationRate = 0.000001;     // Initial probability of mutation
        String nameSurname = "EA"; // Prefix for output filenames

        // Load the target image from file
        BufferedImage target;
        try {
            target = ImageIO.read(new File(targetPath));
            if (target == null) throw new IOException("ImageIO.read returned null");
        } catch (IOException e) {
            e.printStackTrace();
            return; // Exit if image cannot be loaded
        }

        // Create and run the Evolutionary Algorithm
        // System.nanoTime() provides a unique seed for random number generation
        EA ga = new EA(target, populationSize, genesPerIndividual,
                mutationRate, generations, System.nanoTime());
        try {
            // Run the algorithm: imageIndex=1, runNumber=1
            ga.run(nameSurname, 1, 1);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Enum defining the three types of geometric shapes available
    enum ShapeType {TRIANGLE, HEART, ELLIPSE}

    // Represents a single geometric primitive (gene) in the genetic representation
    static class Gene {
        double x = 0, y = 0;           // Position coordinates (0-511)
        int size = 1;                  // Scaling factor (1-6 pixels)
        double angle = 0.0;            // Rotation angle in radians (0-2π)
        int r = 255, g = 0, b = 0;     // RGB color components (0-255)
        double alpha = 0.6;            // Transparency (0.05-1.0)
        ShapeType t = ShapeType.HEART; // Type of shape
        double aspectRatio = 1.0;      // Vertical/horizontal scaling (0.2-3.0)

        // Create a deep copy of this gene
        public Gene copy() {
            Gene ng = new Gene();
            ng.x = x;
            ng.y = y;
            ng.size = size;
            ng.angle = angle;
            ng.r = r;
            ng.g = g;
            ng.b = b;
            ng.alpha = alpha;
            ng.t = t;
            ng.aspectRatio = aspectRatio;
            return ng;
        }
    }

    // Represents a complete image approximation (individual/chromosome)
    static class Individual {
        Gene[] genes;                  // Array of 300 genes (geometric primitives)
        double fitness = -1.0;         // Fitness score (0-1, higher is better)
        boolean evaluated = false;     // Flag to avoid redundant fitness calculations

        // Constructor: creates an individual with specified number of genes
        public Individual(int count) {
            genes = new Gene[count];
            for (int i = 0; i < count; i++) genes[i] = new Gene();
        }

        // Initialize all genes with random values within specified ranges
        void randomize(int width, int height, Random random, EA ea) {
            ShapeType[] allShapes = ShapeType.values();
            for (Gene gene : genes) {
                // Random position within image bounds
                gene.x = random.nextDouble() * width;
                gene.y = random.nextDouble() * height;

                // Random size based on image dimensions (1-6 pixels for 512x512)
                gene.size = 1 + random.nextInt(Math.max(1, Math.min(width, height) / 100));

                // Random rotation angle (0 to 2π)
                gene.angle = random.nextFloat() * (float) Math.PI * 2;

                // Random color centered around global average color ±10
                gene.r = ea.avgR + random.nextInt(21) - 10;
                gene.g = ea.avgG + random.nextInt(21) - 10;
                gene.b = ea.avgB + random.nextInt(21) - 10;
                gene.r = Math.max(0, Math.min(255, gene.r));
                gene.g = Math.max(0, Math.min(255, gene.g));
                gene.b = Math.max(0, Math.min(255, gene.b));

                // Random transparency (0.08 to 0.33)
                gene.alpha = 0.08 + random.nextDouble() * 0.25;

                // Random shape type
                gene.t = allShapes[random.nextInt(allShapes.length)];

                // Random aspect ratio (0.3 to 2.3)
                gene.aspectRatio = 0.3 + random.nextDouble() * 2.0;
            }
            evaluated = false; // Needs fitness evaluation
        }

        // Create a deep copy of this individual
        public Individual copy() {
            Individual c = new Individual(genes.length);
            for (int i = 0; i < genes.length; i++) {
                c.genes[i] = genes[i].copy();
            }
            c.fitness = this.fitness;
            c.evaluated = this.evaluated;
            return c;
        }
    }

    // Represents a population of individuals
    static class Population {
        List<Individual> ind; // List of individuals in the population

        public Population(int capacity) {
            ind = new ArrayList<>(capacity);
        }

        public void add(Individual in) {
            ind.add(in);
        }

        public List<Individual> getInd() {
            return ind;
        }

        public int size() {
            return ind.size();
        }

        // Find the individual with highest fitness in the population
        public Individual getFittest() {
            return ind.stream().max(Comparator.comparingDouble(i -> i.fitness)).orElse(null);
        }
    }

    // Main Evolutionary Algorithm class
    static class EA {
        BufferedImage target;        // Original target image (512x512)
        BufferedImage evalTarget;    // Resized version for faster evaluation (128x128)
        int width = 512;             // Output image width
        int height = 512;            // Output image height
        int evalW = 128;             // Evaluation image width (for speed)
        int evalH = 128;             // Evaluation image height (for speed)
        int populationSize;          // Number of individuals in population
        int genesInd;                // Number of genes per individual (300)
        double crossoverRate = 0.8;  // Probability of crossover (80%)
        double mutationRate;         // Current mutation rate (adaptive)
        int tournamentSize = 8;      // Size of tournament selection
        int maxG;                    // Maximum generations (15000)
        Random random;               // Random number generator
        Population population;       // Current population
        double[] weightMask;         // Weight mask emphasizing center of image
        double[] targetGray;         // Grayscale version of target for edge detection
        double[] targetSobel;        // Sobel edge detection result of target
        int avgR, avgG, avgB;        // Global average color of target image

        // Constructor: initialize algorithm with target image and parameters
        public EA(BufferedImage target, int populationSize, int genesInd,
                  double mutationRate, int maxG, long seed) {
            this.target = target;
            // Create downscaled version for faster fitness evaluation
            this.evalTarget = resizeImage(target, evalW, evalH);
            this.populationSize = populationSize;
            this.genesInd = genesInd;
            this.mutationRate = mutationRate;
            this.maxG = maxG;
            this.random = new Random(seed); // Seed for reproducibility

            // Precompute values needed for fitness evaluation
            this.targetGray = buildGrayArray(this.evalTarget);
            this.targetSobel = computeSobel(this.targetGray, evalW, evalH);
            this.weightMask = buildWeightMask(evalW, evalH);

            // Compute global average color for initialization
            int[] avg = computeGlobalAverageColor(target);
            this.avgR = avg[0];
            this.avgG = avg[1];
            this.avgB = avg[2];
        }

        // Calculate average RGB color of entire image
        int[] computeGlobalAverageColor(BufferedImage img) {
            long r = 0, g = 0, b = 0;
            int width = img.getWidth();
            int height = img.getHeight();
            long total = (long) width * height;

            // Sum all pixel values
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = img.getRGB(x, y);
                    r += (rgb >> 16) & 0xFF;
                    g += (rgb >> 8) & 0xFF;
                    b += rgb & 0xFF;
                }
            }

            // Return average as integer array [R, G, B]
            return new int[]{
                    (int) (r / total),
                    (int) (g / total),
                    (int) (b / total)
            };
        }

        // Create a weight mask that emphasizes the center of the image
        // Center pixels have weight 1.0, edges have weight 0.4
        double[] buildWeightMask(int w, int h) {
            double[] mask = new double[w * h];
            double cx = w / 2.0;
            double cy = h / 2.0;
            double maxDist = Math.hypot(cx, cy);

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    double dx = x - cx;
                    double dy = y - cy;
                    double d = Math.hypot(dx, dy);
                    double norm = d / maxDist; // Normalized distance from center (0-1)
                    double wVal = 1.0 - 0.6 * norm; // Linear falloff
                    if (wVal < 0.4) wVal = 0.4; // Minimum weight at edges
                    mask[y * w + x] = wVal;
                }
            }
            return mask;
        }

        // Resize an image to specified dimensions using bicubic interpolation
        BufferedImage resizeImage(BufferedImage original, int newWidth, int newHeight) {
            BufferedImage resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resized.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.drawImage(original, 0, 0, newWidth, newHeight, null);
            g.dispose();
            return resized;
        }

        // Main evolutionary loop
        public void run(String nameSurname, int imageIndex, int runNumber) throws IOException {
            // Step 1: Initialize population with random individuals
            population = new Population(populationSize);
            for (int i = 0; i < populationSize; i++) {
                Individual ind = new Individual(genesInd);
                ind.randomize(width, height, random, this);
                population.add(ind);
            }

            // Step 2: Evaluate initial population
            evaluatePopulation(population);
            Individual best = population.getFittest().copy();
            double bestFitness = best.fitness;

            // Step 3: Create priority queue to track top 5 individuals
            PriorityQueue<Individual> topIndividuals = new PriorityQueue<>(5,
                    Comparator.comparingDouble(ind -> ind.fitness));
            updateTopList(topIndividuals, best.copy());

            int stagnationCount = 0; // Counter for generations without improvement

            // Step 4: Main evolution loop (15000 generations)
            for (int gen = 1; gen <= maxG; gen++) {
                Population newPop = new Population(populationSize);

                // Elitism: preserve 3 best individuals from previous generation
                newPop.add(best.copy());
                Individual secondBest = tournamentSelection(population);
                Individual thirdBest = tournamentSelection(population);
                newPop.add(secondBest.copy());
                newPop.add(thirdBest.copy());

                // Create remaining individuals through selection and reproduction
                while (newPop.size() < populationSize) {
                    Individual parent1 = tournamentSelection(population);
                    Individual parent2 = tournamentSelection(population);

                    Individual child;
                    if (random.nextDouble() < crossoverRate) {
                        child = crossover(parent1, parent2); // Apply crossover
                    } else {
                        child = parent1.copy(); // Clone parent
                    }

                    mutate(child); // Apply mutation
                    newPop.add(child);
                }

                // Evaluate new generation
                evaluatePopulation(newPop);
                population = newPop;

                // Update best individual if improved
                Individual genBest = population.getFittest();
                if (genBest.fitness > bestFitness) {
                    best = genBest.copy();
                    bestFitness = genBest.fitness;
                    stagnationCount = 0; // Reset stagnation counter
                } else {
                    stagnationCount++; // No improvement this generation
                }

                // Update top individuals list
                updateTopList(topIndividuals, genBest.copy());
                updateTopList(topIndividuals, best.copy());

                // Adaptive mutation: increase rate if stagnating
                if (stagnationCount > 100) {
                    mutationRate = Math.min(0.3, mutationRate * 1.05);
                    stagnationCount = 0; // Reset after adjustment
                }
            }

            // Step 5: Get final best individuals (top 5)
            List<Individual> finalBestIndividuals = getFinalBestIndividuals(topIndividuals, best);

            // Step 6: Save results to files
            saveBestIndividuals(finalBestIndividuals, nameSurname, imageIndex, runNumber);
        }

        // Two-point crossover: combines genes from two parents
        Individual crossover(Individual a, Individual b) {
            // Select fitter parent as "better"
            Individual better = (a.fitness > b.fitness) ? a : b;
            Individual worse = (a.fitness > b.fitness) ? b : a;

            Individual child = new Individual(genesInd);

            // Select two random crossover points
            int point1 = random.nextInt(genesInd / 3);
            int point2 = point1 + random.nextInt(genesInd / 3) + genesInd / 3;

            // Copy genes: mostly from better parent, middle section from worse parent
            for (int i = 0; i < genesInd; i++) {
                if (i < point1 || i >= point2) {
                    // Outside crossover region: take from better parent
                    child.genes[i] = better.genes[i].copy();
                } else {
                    // Inside crossover region: take from worse parent
                    child.genes[i] = worse.genes[i].copy();
                }
            }

            child.evaluated = false; // Needs fitness evaluation
            return child;
        }

        // Apply mutation to an individual
        void mutate(Individual ind) {
            // Adaptive mutation rate: higher for less fit individuals
            double individualMutationRate = mutationRate;
            if (ind.fitness < 0.7) {
                individualMutationRate *= 1.5;
            }

            // Calculate number of genes to mutate
            int genesToMutate = Math.max(1, (int) (individualMutationRate * ind.genes.length));

            // Mutate random genes
            for (int i = 0; i < genesToMutate; i++) {
                int geneIndex = random.nextInt(ind.genes.length);
                mutateGene(ind.genes[geneIndex]);
            }

            ind.evaluated = false; // Fitness needs recalculation
        }

        // Apply one of 8 possible mutations to a single gene
        void mutateGene(Gene gene) {
            int choice = random.nextInt(8);

            if (choice == 0) {
                // Mutate X position: Gaussian noise ±1% of width
                gene.x = clamp(gene.x + random.nextGaussian() * width * 0.01, 0, width - 1);
            }
            else if (choice == 1) {
                // Mutate Y position: Gaussian noise ±1% of height
                gene.y = clamp(gene.y + random.nextGaussian() * height * 0.01, 0, height - 1);
            }
            else if (choice == 2) {
                // Mutate size: Gaussian noise ±2 pixels
                gene.size = clampInt(gene.size + (int)Math.round(random.nextGaussian() * 2),
                        1, Math.max(2, Math.min(width, height)/8));
            }
            else if (choice == 3) {
                // Mutate rotation angle: Gaussian noise ±0.2 radians
                gene.angle = (gene.angle + random.nextGaussian() * 0.2) % (2*Math.PI);
            }
            else if (choice == 4) {
                // Mutate color: uniform change ±10 for each RGB component
                gene.r = clampInt(gene.r + random.nextInt(21) - 10, 0, 255);
                gene.g = clampInt(gene.g + random.nextInt(21) - 10, 0, 255);
                gene.b = clampInt(gene.b + random.nextInt(21) - 10, 0, 255);
            }
            else if (choice == 5) {
                // Mutate transparency: Gaussian noise ±0.04
                gene.alpha = clampDouble(gene.alpha + random.nextGaussian()*0.04, 0.05, 1.0);
            }
            else if (choice == 6) {
                // Change shape type randomly
                ShapeType[] shapes = ShapeType.values();
                gene.t = shapes[random.nextInt(shapes.length)];
            }
            else {
                // Mutate aspect ratio: Gaussian noise ±0.2
                gene.aspectRatio = clampDouble(gene.aspectRatio + random.nextGaussian() * 0.2, 0.2, 3.0);
            }
        }

        // Collect top 5 best individuals from evolution
        List<Individual> getFinalBestIndividuals(PriorityQueue<Individual> topQueue, Individual globalBest) {
            List<Individual> result = new ArrayList<>();

            // Always include the global best
            result.add(globalBest.copy());

            // Get top individuals from priority queue (sorted by fitness)
            List<Individual> topList = new ArrayList<>(topQueue);
            topList.sort((a, b) -> Double.compare(b.fitness, a.fitness));

            // Add unique high-fitness individuals
            for (Individual ind : topList) {
                if (result.size() >= 5) break;

                // Check for duplicates (same fitness)
                boolean isDuplicate = false;
                for (Individual existing : result) {
                    if (Math.abs(existing.fitness - ind.fitness) < 0.0001) {
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate) {
                    result.add(ind.copy());
                }
            }

            // If we don't have 5 unique individuals, create variants of global best
            while (result.size() < 5) {
                Individual mutated = globalBest.copy();

                // Apply 5 random mutations to create variation
                for (int i = 0; i < 5; i++) {
                    mutateGene(mutated.genes[random.nextInt(mutated.genes.length)]);
                }
                mutated.evaluated = false;
                evaluateIndividual(mutated);
                result.add(mutated);
            }

            return result;
        }

        // Save best individuals as JPEG files
        void saveBestIndividuals(List<Individual> bestIndividuals, String nameSurname,
                                 int imageIndex, int runNumber) throws IOException {


            for (int i = 0; i < bestIndividuals.size(); i++) {
                Individual ind = bestIndividuals.get(i);
                BufferedImage img = renderIndividual(ind); // Render full 512x512 image
                String filename = String.format("%sOutput%d_%d_%d.jpg",
                        nameSurname, imageIndex, runNumber, i + 1);

                ImageIO.write(img, "jpg", new File(filename));
            }
        }

        // Update priority queue with top 5 individuals
        void updateTopList(PriorityQueue<Individual> top, Individual candidate) {
            if (top.size() < 5) {
                top.offer(candidate); // Add if queue not full
            } else if (candidate.fitness > top.peek().fitness) {
                top.poll(); // Remove lowest
                top.offer(candidate); // Add new candidate
            }
        }

        // Evaluate all individuals in population (parallel execution)
        void evaluatePopulation(Population pop) {
            pop.getInd().parallelStream().forEach(ind -> {
                if (!ind.evaluated) evaluateIndividual(ind);
            });
        }

        // Calculate fitness for a single individual
        void evaluateIndividual(Individual ind) {
            if (ind.evaluated) return; // Skip if already evaluated
            BufferedImage rendered = renderIndividualEval(ind); // Render at 128x128
            ind.fitness = computeFitness(rendered); // Calculate fitness
            ind.evaluated = true;
        }

        // Calculate fitness score (0-1) comparing rendered image to target
        double computeFitness(BufferedImage rendered) {
            int w = evalW, h = evalH;
            int n = w * h; // Total pixels

            // Get pixels from rendered image
            int[] renderedPixels = new int[n];
            rendered.getRGB(0, 0, w, h, renderedPixels, 0, w);

            long errColor = 0; // Accumulated color error

            // Calculate weighted color difference (MSE)
            for (int i = 0; i < n; i++) {
                // Target pixel color
                int rgbT = evalTarget.getRGB(i % w, i / w);
                int rT = (rgbT >> 16) & 0xFF;
                int gT = (rgbT >> 8) & 0xFF;
                int bT =  rgbT & 0xFF;

                // Rendered pixel color
                int rgbR = renderedPixels[i];
                int rR = (rgbR >> 16) & 0xFF;
                int gR = (rgbR >> 8) & 0xFF;
                int bR =  rgbR & 0xFF;

                // Color differences
                int dr = rT - rR;
                int dg = gT - gR;
                int db = bT - bR;

                // Weighted squared error
                double wc = weightMask[i];
                errColor += (long)((dr*dr + dg*dg + db*db) * wc);
            }

            // Convert to Mean Squared Error and normalize to 0-1 range
            double mseC = (double)errColor / n;
            double colorScore = 1.0 / (1.0 + Math.sqrt(mseC) / 255.0);

            // Edge preservation component
            int[] rp = renderedPixels;
            double[] gray = buildGrayArrayFromRGB(rp, w, h);
            double[] sobel = computeSobel(gray, w, h);

            double errS = 0;
            for (int i = 0; i < n; i++) {
                double d = targetSobel[i] - sobel[i];
                errS += d*d * weightMask[i];
            }

            double mseS = errS / n;
            double sobelScore = 1.0 / (1.0 + Math.sqrt(mseS)/255.0);

            // Combined fitness: 70% color similarity, 30% edge preservation
            return 0.7 * colorScore + 0.3 * sobelScore;
        }

        // Render individual at evaluation resolution (128x128) for fitness calculation
        BufferedImage renderIndividualEval(Individual ind) {
            BufferedImage canvas = new BufferedImage(evalW, evalH, BufferedImage.TYPE_INT_RGB);
            Graphics2D g2 = canvas.createGraphics();
            g2.setColor(Color.WHITE);
            g2.fillRect(0, 0, evalW, evalH);

            // Enable anti-aliasing for smoother shapes
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);

            // Scaling factors from full resolution to evaluation resolution
            double sx = evalW / (double) width;
            double sy = evalH / (double) height;

            // Draw shapes from back to front (reverse order for proper blending)
            for (int i = ind.genes.length - 1; i >= 0; i--) {
                drawShapeScaled(g2, ind.genes[i], sx, sy);
            }
            g2.dispose();
            return canvas;
        }

        // Draw a single shape at evaluation scale
        void drawShapeScaled(Graphics2D g2, Gene gene, double sx, double sy) {
            AffineTransform old = g2.getTransform();
            Composite oldComp = g2.getComposite();

            // Scale position from 512x512 to 128x128
            double cx = gene.x * sx;
            double cy = gene.y * sy;
            double size = gene.size * Math.max(sx, sy);

            // Apply transformations: translate, rotate, scale
            g2.translate(cx, cy);
            g2.rotate(gene.angle);
            double scale = size / 6.0;
            g2.scale(scale, scale * gene.aspectRatio);

            // Set transparency (alpha blending)
            float alpha = (float) Math.max(0.01, Math.min(1.0, gene.alpha));
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
            g2.setPaint(new Color(gene.r, gene.g, gene.b));

            // Create and fill the shape
            Shape s = createShape(gene);
            g2.fill(s);

            // Restore original graphics state
            g2.setTransform(old);
            g2.setComposite(oldComp);
        }

        // Render individual at full resolution (512x512) for final output
        BufferedImage renderIndividual(Individual ind) {
            BufferedImage canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            Graphics2D g2 = canvas.createGraphics();
            g2.setColor(Color.WHITE);
            g2.fillRect(0, 0, width, height);
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);

            // Draw all shapes from back to front
            for (int i = ind.genes.length - 1; i >= 0; i--) {
                drawShapeFull(g2, ind.genes[i]);
            }
            g2.dispose();
            return canvas;
        }

        // Draw a single shape at full resolution
        void drawShapeFull(Graphics2D g2, Gene gene) {
            AffineTransform old = g2.getTransform();
            Composite oldComp = g2.getComposite();

            // Apply transformations
            g2.translate(gene.x, gene.y);
            g2.rotate(gene.angle);
            double scale = gene.size / 6.0;
            g2.scale(scale, scale * gene.aspectRatio);

            // Set transparency and color
            float alpha = (float) Math.max(0.01, Math.min(1.0, gene.alpha));
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
            g2.setPaint(new Color(gene.r, gene.g, gene.b));

            // Create and fill shape
            Shape s = createShape(gene);
            g2.fill(s);

            // Restore graphics state
            g2.setTransform(old);
            g2.setComposite(oldComp);
        }

        // Create geometric shape based on gene type
        Shape createShape(Gene gene) {
            switch (gene.t) {
                case TRIANGLE:
                    GeneralPath tri = new GeneralPath();
                    tri.moveTo(0, -9);
                    tri.lineTo(8, 8);
                    tri.lineTo(-8, 8);
                    tri.closePath();
                    return tri;
                case HEART:
                    return makeHeartShape();
                case ELLIPSE:
                    return new Ellipse2D.Double(-6, -4, 12, 8);
                default:
                    return new Ellipse2D.Double(-6, -6, 12, 12);
            }
        }

        // Tournament selection: pick best from random sample
        Individual tournamentSelection(Population pop) {
            Individual best = null;
            for (int i = 0; i < tournamentSize; i++) {
                Individual cand = pop.getInd().get(random.nextInt(pop.size()));
                if (best == null || cand.fitness > best.fitness) best = cand;
            }
            return best.copy();
        }

        // Utility functions for clamping values to ranges
        double clamp(double v, double min, double max) { return Math.max(min, Math.min(max, v)); }
        int clampInt(int v, int min, int max) { return Math.max(min, Math.min(max, v)); }
        double clampDouble(double v, double min, double max) { return Math.max(min, Math.min(max, v)); }

        // Convert image to grayscale array
        double[] buildGrayArray(BufferedImage img) {
            int w = img.getWidth(), h = img.getHeight();
            double[] gray = new double[w * h];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int rgb = img.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    // Standard luminance formula
                    gray[y * w + x] = 0.299 * r + 0.587 * g + 0.114 * b;
                }
            }
            return gray;
        }

        // Convert RGB pixel array to grayscale
        double[] buildGrayArrayFromRGB(int[] pixels, int w, int h) {
            double[] gray = new double[w * h];
            for (int i = 0; i < pixels.length; i++) {
                int rgb = pixels[i];
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
            return gray;
        }

        // Compute Sobel edge detection filter
        double[] computeSobel(double[] gray, int w, int h) {
            double[] sob = new double[w * h];

            // Skip borders (no neighbors for edge detection)
            for (int y = 1; y < h - 1; y++) {
                for (int x = 1; x < w - 1; x++) {
                    int i = y * w + x;
                    // Horizontal gradient (difference between right and left neighbors)
                    double gx = gray[i + 1] - gray[i - 1];
                    // Vertical gradient (difference between bottom and top neighbors)
                    double gy = gray[i + w] - gray[i - w];
                    // Combined gradient magnitude
                    sob[i] = Math.abs(gx) + Math.abs(gy);
                }
            }
            return sob;
        }

        // Create heart shape using Bezier curves
        Shape makeHeartShape() {
            GeneralPath p = new GeneralPath();
            p.moveTo(0, -1.0);
            p.curveTo(3, -12, 14, -7, 14, 3);
            p.curveTo(14, 14, 7, 22, 0, 28);
            p.curveTo(-7, 22, -14, 14, -14, 3);
            p.curveTo(-14, -7, -3, -12, 0, -5);
            p.closePath();
            return p;
        }
    }
}