#define MAX_LINE_POINTS 512
#define JM_THRESHOLD 0.9f

/* Get line opencl kernel version */
void get_line(int x0, int y0, int x1, int y1, int *points, int *numOfDots)
{
    int x;
    int y;
    int dx;
    int dy;
    int sx = 1;
    int sy = 1;
    float err = 0.0f;

    numOfDots[0] = 0;

    dx = abs(x1 - x0);
    dy = abs(y1 - y0);
    x = x0;
    y = y0;

    if (x0 > x1)
    {
        sx = -1;
    }

    if (y0 > y1)
    {
        sy = -1;
    }

    if (dx > dy)
    {
        err = dx / 2.0f;
        while (x != x1)
        {
            points[numOfDots[0] * 2] = x;
            points[numOfDots[0] * 2 + 1] = y;
            numOfDots[0]++;
            err -= dy;
            if (err < 0.0f)
            {
                y += sy;
                err += dx;
            }
            x += sx;
        }
    }
    else
    {
        err = dy / 2.0f;
        while (y != y1)
        {
            points[numOfDots[0] * 2] = x;
            points[numOfDots[0] * 2 + 1] = y;
            numOfDots[0]++;
            err -= dx;
            if (err < 0.0f)
            {
                x += sx;
                err += dy;
            }
            y += sy;
        }
    }

    points[numOfDots[0] * 2] = x;
    points[numOfDots[0] * 2 + 1] = y;
    numOfDots[0]++;
}

__kernel void getPairScore(__global float const *pts1, __global float const *pts2, __global float const *jm_x, __global float const *jm_y, __global int const *pts_len, __global int const *jm_len, __global float *pairs)
{
    /* m and n in pts1 and pts2 */
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);

    /* 2 points */
    float x1;
    float y1;
    float x2;
    float y2;

    float dx;
    float dy;
    float norm;
    float ux;
    float uy;
    int rr;
    int cc;
    float paf_score = 0.0f;
    float paf_score_total = 0.0f;
    int paf_score_thresh_count = 0.0f;
    float score;
    float correct_ratio = 0.0f;
    int line_points[MAX_LINE_POINTS * 2];
    int numOfDots = 0;

    x1 = pts1[i * 2];
    y1 = pts1[i * 2 + 1];
    x2 = pts2[j * 2];
    y2 = pts2[j * 2 + 1];

    get_line((int)x1, (int)y1, (int)x2, (int)y2, line_points, &numOfDots);
    dx = x2 - x1;
    dy = y2 - y1;
    norm = sqrt((dx * dx) + (dy * dy));

    if (norm > 0)
    {
        ux = dx / norm;
        uy = dy / norm;

        for (int k = 0; k < numOfDots; k++)
        {
            rr = line_points[k * 2];
            cc = line_points[k * 2 + 1];
            paf_score = (jm_x[cc * jm_len[1] + rr] * ux) + (jm_y[cc * jm_len[1] + rr] * uy);
            paf_score_total += paf_score;

            if (paf_score > 0.4f)
            {
                paf_score_thresh_count++;
            }
        }

        score = paf_score_total / numOfDots;
        correct_ratio = (float)paf_score_thresh_count / (float)numOfDots;

        if (correct_ratio < JM_THRESHOLD)
        {
        }
        else
        {
            pairs[(i * pts_len[1]) + j] = score * correct_ratio;
        }
    }
    else
    {
    }
}
