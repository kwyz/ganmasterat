export default {
    name: "chart-component",
    poprs: {
        data: Array
    },
    data: () => {
        return {
            chartConfig: {
                type: "bar",
                dataset: {
                    label: 'Generation scale',
                    data: [],
                    borderWidth: 1
                },
                options: {
                    scales: {
                        yAxes: [{
                            ticks: {
                                beginAtZero: true
                            }
                        }]
                    }
                }
            },
            surface: null,
        };
    },
    created() {},
    methods: {

    }
}